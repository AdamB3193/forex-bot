"""
Download daily OHLCV data for all 24 pairs from Dukascopy and persist to
data\raw\{PAIR}.csv and the raw_ohlcv SQLite table.

Usage:
    python download.py              # skip pairs that already have a CSV
    python download.py --force      # re-download everything
    python download.py --from-local # load existing CSVs into DB, no network
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

# Allow running this file directly from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_END_DATE,
    DATA_START_DATE,
    MIN_BARS_REQUIRED,
    PAIRS,
    RAW_DATA_DIR,
)
from database import get_connection, insert_ohlcv_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_SLEEP = 5  # seconds between retries

# Dukascopy free API requires EUR/USD style (slash after base currency)
def _to_api_symbol(pair: str) -> str:
    return pair[:3] + '/' + pair[3:]


def _normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise whatever dukascopy-python returns to date, open, high, low, close, tick_volume."""
    df = raw_df.copy()

    # --- index → date column ---
    if df.index.name in ('timestamp', 'time') or pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, utc=True).normalize().strftime('%Y-%m-%d')
        df.index.name = 'date'
    df = df.reset_index()

    # Rename index column if it came through as 'timestamp' or 'time'
    for col in ('timestamp', 'time'):
        if col in df.columns:
            df.rename(columns={col: 'date'}, inplace=True)

    # Lowercase all columns
    df.columns = [c.lower() for c in df.columns]

    # volume → tick_volume
    for alias in ('volume', 'Volume', 'vol'):
        if alias.lower() in df.columns and 'tick_volume' not in df.columns:
            df.rename(columns={alias.lower(): 'tick_volume'}, inplace=True)
            break
    if 'tick_volume' not in df.columns:
        df['tick_volume'] = 0.0

    # Ensure date is a plain YYYY-MM-DD string (strip time component)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.strftime('%Y-%m-%d')

    # Keep only the columns we care about
    keep = [c for c in ('date', 'open', 'high', 'low', 'close', 'tick_volume') if c in df.columns]
    df = df[keep].copy()

    # Drop duplicate dates (e.g. Sunday micro-stubs will be handled by clean.py)
    df = df.drop_duplicates(subset='date', keep='last')
    df = df.sort_values('date').reset_index(drop=True)

    return df


def _checksum(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _record_version(conn, pair: str, df: pd.DataFrame, rows_inserted: int) -> None:
    conn.execute(
        """
        INSERT INTO data_versions (downloaded_at, source, pair, date_from, date_to, rows_inserted, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            'dukascopy_freeserv',
            pair,
            df['date'].min(),
            df['date'].max(),
            rows_inserted,
            _checksum(df),
        ),
    )
    conn.commit()


def _fetch_pair(pair: str) -> pd.DataFrame:
    """Download one pair from Dukascopy with retry logic. Returns normalised DataFrame."""
    try:
        import dukascopy_python
    except ImportError:
        _print_manual_fallback()
        sys.exit(1)

    api_symbol = _to_api_symbol(pair)
    start_dt = datetime.fromisoformat(DATA_START_DATE)
    end_dt   = datetime.fromisoformat(DATA_END_DATE)

    last_exc = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            raw = dukascopy_python.fetch(
                instrument=api_symbol,
                interval=dukascopy_python.INTERVAL_DAY_1,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start_dt,
                end=end_dt,
                max_retries=5,
            )
            df = _normalize_df(raw)
            log.info('Downloaded %s: %d rows from %s to %s',
                     pair, len(df), df['date'].min(), df['date'].max())
            return df
        except Exception as exc:
            last_exc = exc
            log.warning('Attempt %d/%d failed for %s: %s', attempt, _RETRY_ATTEMPTS, pair, exc)
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_SLEEP)

    raise RuntimeError(f'All {_RETRY_ATTEMPTS} attempts failed for {pair}') from last_exc


def _load_from_csv(pair: str) -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, f'{pair}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No CSV found at {path}')
    df = pd.read_csv(path)
    return _normalize_df(df)


def _save_to_csv(pair: str, df: pd.DataFrame) -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, f'{pair}.csv')
    df.to_csv(path, index=False)
    log.info('Saved raw CSV → %s', path)


def _insert_and_record(pair: str, df: pd.DataFrame) -> int:
    """Set the index to 'date' so insert_ohlcv_batch picks it up, then record the version."""
    df_indexed = df.set_index('date')
    rows = insert_ohlcv_batch(pair, df_indexed)
    with get_connection() as conn:
        _record_version(conn, pair, df, rows)
    return rows


def download_all_pairs(force_redownload: bool = False, from_local: bool = False) -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    results = {}   # pair → {'rows': int, 'date_from': str, 'date_to': str, 'status': str}

    for pair in PAIRS:
        csv_path = os.path.join(RAW_DATA_DIR, f'{pair}.csv')
        csv_exists = os.path.exists(csv_path)

        try:
            if from_local:
                log.info('[%s] Loading from local CSV …', pair)
                df = _load_from_csv(pair)
            elif csv_exists and not force_redownload:
                log.info('[%s] CSV exists, loading from disk (use --force to re-download)', pair)
                df = _load_from_csv(pair)
            else:
                df = _fetch_pair(pair)
                _save_to_csv(pair, df)

            rows = _insert_and_record(pair, df)
            results[pair] = {
                'rows':      rows,
                'date_from': df['date'].min(),
                'date_to':   df['date'].max(),
                'status':    'OK',
            }

        except Exception as exc:
            log.error('[%s] FAILED: %s', pair, exc)
            results[pair] = {'rows': 0, 'date_from': '–', 'date_to': '–', 'status': f'ERROR: {exc}'}

    _print_summary(results)


def _print_summary(results: dict) -> None:
    print()
    print('=' * 80)
    print(f'{"PAIR":<10}  {"ROWS":>6}  {"FROM":>12}  {"TO":>12}  {"STATUS"}')
    print('-' * 80)
    for pair, r in results.items():
        flag = '' if r['rows'] >= MIN_BARS_REQUIRED else '  *** NEEDS ATTENTION'
        print(f'{pair:<10}  {r["rows"]:>6}  {r["date_from"]:>12}  {r["date_to"]:>12}  {r["status"]}{flag}')
    print('=' * 80)
    ok   = sum(1 for r in results.values() if r['status'] == 'OK')
    fail = len(results) - ok
    low  = sum(1 for r in results.values() if 0 < r['rows'] < MIN_BARS_REQUIRED)
    print(f'  {ok} succeeded  |  {fail} failed  |  {low} below MIN_BARS_REQUIRED ({MIN_BARS_REQUIRED})')
    print()


def _print_manual_fallback() -> None:
    root = os.path.join('C:\\Users\\Adam\\OneDrive\\Documents\\Projects\\Forex Bot')
    print()
    print('dukascopy-python API has changed. Manual download fallback:')
    print('  1. Go to https://www.histdata.com/download-free-forex-historical-data/?/ascii/daily-candles/')
    print('  2. Download each pair as CSV')
    print(f'  3. Save to {root}\\data\\raw\\{{PAIR}}.csv')
    print('  4. Run: python download.py --from-local')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Dukascopy OHLCV data for all 24 pairs')
    parser.add_argument('--force',      action='store_true', help='Re-download even if CSV already exists')
    parser.add_argument('--from-local', action='store_true', dest='from_local',
                        help='Load existing CSVs into DB without hitting the network')
    args = parser.parse_args()

    download_all_pairs(force_redownload=args.force, from_local=args.from_local)
