"""
Master orchestrator for the forex bot data pipeline.

Usage:
    python pipeline.py --full       # Complete initial build from scratch
    python pipeline.py --update     # Weekly incremental update (new bars only)
    python pipeline.py --validate   # Run validation report only
    python pipeline.py --status     # Print current DB status table
"""

import argparse
import bisect
import logging
import os
import sys
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, os.path.dirname(__file__))

from config import DB_PATH, LOGS_DIR, MIN_BARS_REQUIRED, PAIRS, SPREADS
from database import get_connection, init_database, insert_ohlcv_batch
from download import _normalize_df, _to_api_symbol, download_all_pairs
from clean import clean_all_pairs, detect_anomalies
from timezone_fix import remove_sunday_stubs
from validate import run_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_SLEEP    = 5


# ---------------------------------------------------------------------------
# Tee writer — mirrors stdout to a log file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file   = open(path, 'w', encoding='utf-8')
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._file.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


# ---------------------------------------------------------------------------
# Phase timing helpers
# ---------------------------------------------------------------------------

def _phase_start(label: str) -> float:
    print(f'\n{"=" * 70}')
    print(f'  {label}')
    print(f'{"=" * 70}')
    return time.time()


def _phase_end(label: str, t0: float) -> float:
    elapsed = time.time() - t0
    print(f'\n  [{label}] completed in {elapsed:.1f}s')
    return elapsed


# ---------------------------------------------------------------------------
# Full initial pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> None:
    """Complete pipeline: init -> download -> clean -> validate."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ts       = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f'pipeline_run_{ts}.txt')
    tee      = _Tee(log_path)
    _orig    = sys.stdout
    sys.stdout = tee

    t_wall  = time.time()
    timings = {}

    try:
        print(f'\n{"=" * 70}')
        print(f'  FOREX BOT DATA PIPELINE -- FULL RUN')
        print(f'  Started : {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}')
        print(f'{"=" * 70}')

        t = _phase_start('PHASE 1/4  INIT DATABASE')
        init_database()
        timings['init'] = _phase_end('init', t)

        t = _phase_start('PHASE 2/4  DOWNLOAD DATA')
        download_all_pairs(force_redownload=False)
        timings['download'] = _phase_end('download', t)

        t = _phase_start('PHASE 3/4  CLEAN DATA')
        clean_all_pairs()
        timings['clean'] = _phase_end('clean', t)

        t = _phase_start('PHASE 4/4  VALIDATE')
        run_validation()
        timings['validate'] = _phase_end('validate', t)

        total = time.time() - t_wall
        print(f'\n{"=" * 70}')
        print(f'  PIPELINE COMPLETE')
        print(f'  Total    : {total:.1f}s')
        print(
            f'  init={timings["init"]:.1f}s  '
            f'download={timings["download"]:.1f}s  '
            f'clean={timings["clean"]:.1f}s  '
            f'validate={timings["validate"]:.1f}s'
        )
        print(f'{"=" * 70}\n')

    finally:
        sys.stdout = _orig
        tee.close()

    print(f'Pipeline log saved -> {log_path}')


# ---------------------------------------------------------------------------
# Weekly incremental update — helpers
# ---------------------------------------------------------------------------

def _fetch_new_bars(pair: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily bars for *pair* between start_date and end_date (inclusive)."""
    try:
        import dukascopy_python
    except ImportError:
        raise RuntimeError(
            'dukascopy-python not installed. Run: pip install dukascopy-python'
        )

    import time as _t
    api_symbol = _to_api_symbol(pair)
    start_dt   = datetime.fromisoformat(start_date)
    end_dt     = datetime.fromisoformat(end_date)
    last_exc   = None

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
            return _normalize_df(raw)
        except Exception as exc:
            last_exc = exc
            log.warning('[%s] attempt %d/%d failed: %s', pair, attempt, _RETRY_ATTEMPTS, exc)
            if attempt < _RETRY_ATTEMPTS:
                _t.sleep(_RETRY_SLEEP)

    raise RuntimeError(f'All {_RETRY_ATTEMPTS} attempts failed for {pair}') from last_exc


def _atr_incremental(df: pd.DataFrame, last_close: float,
                     last_atr_14: float, last_atr_200: float) -> pd.DataFrame:
    """Extend Wilder ATR forward for *df* (real bars only) using last known values as seed."""
    df  = df.copy()
    n   = len(df)
    hi  = df['high'].to_numpy(dtype=float)
    lo  = df['low'].to_numpy(dtype=float)
    cl  = df['close'].to_numpy(dtype=float)

    prev_cl    = np.empty(n, dtype=float)
    prev_cl[0] = last_close
    if n > 1:
        prev_cl[1:] = cl[:-1]

    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))

    a14, a200 = 1.0 / 14, 1.0 / 200
    atr14     = np.empty(n, dtype=float)
    atr200    = np.empty(n, dtype=float)
    p14, p200 = last_atr_14, last_atr_200

    for i in range(n):
        p14      = p14  * (1.0 - a14)  + tr[i] * a14
        p200     = p200 * (1.0 - a200) + tr[i] * a200
        atr14[i]  = p14
        atr200[i] = p200

    df['atr_14']  = atr14
    df['atr_200'] = atr200
    return df


def _fill_new_gaps(df: pd.DataFrame, prev_date: str, prev_close: float) -> pd.DataFrame:
    """Insert forward-filled rows for missing business days in *df*'s date range."""
    if df.empty:
        return df

    start    = (pd.Timestamp(prev_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end      = df['date'].max()
    expected = set(pd.date_range(start=start, end=end, freq=CustomBusinessDay()).strftime('%Y-%m-%d'))
    actual   = set(df['date'])
    missing  = sorted(expected - actual)

    if not missing:
        return df

    all_dates = sorted([prev_date] + df['date'].tolist())
    close_map = {prev_date: prev_close}
    close_map.update(dict(zip(df['date'], df['close'])))

    gap_rows = []
    for gap_date in missing:
        idx = bisect.bisect_left(all_dates, gap_date) - 1
        if idx < 0:
            continue
        pc = close_map.get(all_dates[idx], prev_close)
        gap_rows.append({
            'date': gap_date, 'open': pc, 'high': pc, 'low': pc,
            'close': pc, 'tick_volume': 0.0,
            'is_gap': 1, 'is_anomaly': 0, 'is_sunday_stub': 0,
        })

    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        log.info('Weekly update: filled %d gap bars', len(gap_rows))

    return df


# ---------------------------------------------------------------------------
# Weekly incremental update
# ---------------------------------------------------------------------------

def run_weekly_update() -> None:
    """Download and upsert only bars newer than the latest DB date, for all pairs."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    today           = date.today().isoformat()
    n_new_total     = 0
    n_pairs_updated = 0

    print(f'\n{"=" * 70}')
    print(f'  WEEKLY UPDATE  ({today})')
    print(f'{"=" * 70}')

    for pair in PAIRS:
        try:
            with get_connection() as conn:
                # Latest date in DB — determines download start
                r1 = conn.execute(
                    'SELECT MAX(date) FROM raw_ohlcv WHERE pair = ?', (pair,)
                ).fetchone()
                latest_date = r1[0] if r1 else None

                if latest_date is None:
                    log.warning('[%s] No data in DB -- run --full first', pair)
                    continue

                # Last real (non-gap) bar for ATR seed
                r2 = conn.execute(
                    'SELECT close, atr_14, atr_200 FROM raw_ohlcv '
                    'WHERE pair = ? AND is_gap = 0 '
                    'ORDER BY date DESC LIMIT 1',
                    (pair,),
                ).fetchone()

            if r2 is None or r2[1] is None or r2[2] is None:
                log.warning('[%s] NULL ATR in DB -- run --full to reprocess', pair)
                continue

            last_close, last_atr_14, last_atr_200 = r2

            next_date = (
                pd.Timestamp(latest_date) + pd.Timedelta(days=1)
            ).strftime('%Y-%m-%d')

            if next_date > today:
                print(f'  [{pair:10s}]  already current ({latest_date})')
                continue

            # Download new bars
            new_df = _fetch_new_bars(pair, next_date, today)

            # Strip weekend stubs from the fresh download
            new_df = remove_sunday_stubs(new_df)

            # Keep only dates strictly after what is already in the DB
            new_df = new_df[new_df['date'] > latest_date].reset_index(drop=True)

            if new_df.empty:
                print(f'  [{pair:10s}]  no new bars (current through {latest_date})')
                continue

            # Initialise flag columns
            for col, val in (('is_gap', 0), ('is_anomaly', 0), ('is_sunday_stub', 0)):
                if col not in new_df.columns:
                    new_df[col] = val

            # Forward-fill any missing business days in the new range
            new_df = _fill_new_gaps(new_df, latest_date, last_close)

            # ATR: extend Wilder EWM on real bars only, then ffill to gap bars
            new_df['atr_14']  = np.nan
            new_df['atr_200'] = np.nan
            real_mask = new_df['is_gap'] == 0
            if real_mask.any():
                real_df = new_df[real_mask].copy().reset_index(drop=True)
                real_df = _atr_incremental(real_df, last_close, last_atr_14, last_atr_200)
                new_df.loc[real_mask, 'atr_14']  = real_df['atr_14'].to_numpy()
                new_df.loc[real_mask, 'atr_200'] = real_df['atr_200'].to_numpy()
            new_df['atr_14']  = new_df['atr_14'].ffill()
            new_df['atr_200'] = new_df['atr_200'].ffill()

            # Anomaly detection on new bars (requires atr_14)
            new_df = detect_anomalies(pair, new_df)

            # Tick-volume z-score using the existing population mu/sigma from DB
            with get_connection() as conn:
                stats = conn.execute(
                    """
                    SELECT AVG(tick_volume),
                           AVG(tick_volume * tick_volume) - AVG(tick_volume) * AVG(tick_volume)
                    FROM raw_ohlcv
                    WHERE pair = ? AND is_gap = 0
                    """,
                    (pair,),
                ).fetchone()
            mu       = float(stats[0] or 0.0)
            variance = float(max(stats[1] or 0.0, 0.0))
            sigma    = variance ** 0.5

            if sigma > 0:
                new_df['tick_vol_zscore'] = (new_df['tick_volume'] - mu) / sigma
            else:
                new_df['tick_vol_zscore'] = 0.0
            new_df.loc[new_df['is_gap'] == 1, 'tick_vol_zscore'] = -3.0

            # All new bars belong to the test split (post-2023)
            new_df['split_set']  = 'test'
            new_df['spread_est'] = SPREADS.get(pair, 0.0)

            rows = insert_ohlcv_batch(pair, new_df.set_index('date'))
            n_new_total     += rows
            n_pairs_updated += 1
            print(
                f'  [{pair:10s}]  +{rows} bars  '
                f'({latest_date} -> {new_df["date"].max()})'
            )

        except Exception as exc:
            import traceback
            log.error('[%s] Update failed: %s', pair, exc)
            traceback.print_exc()

    print(f'\n{"=" * 70}')
    print(
        f'  Weekly update complete: '
        f'{n_new_total} new bars added across {n_pairs_updated} pairs'
    )
    print(f'{"=" * 70}\n')


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def print_status() -> None:
    """Print a per-pair status table from the live DB."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                pair,
                SUM(CASE WHEN split_set = 'train' THEN 1 ELSE 0 END) AS train,
                SUM(CASE WHEN split_set = 'val'   THEN 1 ELSE 0 END) AS val,
                SUM(CASE WHEN split_set = 'test'  THEN 1 ELSE 0 END) AS test,
                COUNT(*)                                               AS total,
                MIN(date)                                              AS date_from,
                MAX(date)                                              AS date_to
            FROM raw_ohlcv
            GROUP BY pair
            ORDER BY pair
            """,
        ).fetchall()

    if not rows:
        print('Database is empty. Run: python pipeline.py --full')
        return

    total_bars = sum(r[4] for r in rows)

    print(f'\n{"=" * 78}')
    print(
        f'  DATABASE STATUS  '
        f'({datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})'
    )
    print(f'  {DB_PATH}')
    print(f'{"=" * 78}')
    sep = f'  {"-"*10}  {"-"*6}  {"-"*5}  {"-"*5}  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*6}'
    print(
        f'  {"PAIR":<10}  {"TRAIN":>6}  {"VAL":>5}  {"TEST":>5}  {"TOTAL":>6}  '
        f'{"FROM":>12}  {"TO":>12}  STATUS'
    )
    print(sep)

    for pair, train, val, test, total, date_from, date_to in rows:
        status = 'OK' if total >= MIN_BARS_REQUIRED else 'LOW'
        print(
            f'  {pair:<10}  {train:>6}  {val:>5}  {test:>5}  {total:>6}  '
            f'{date_from:>12}  {date_to:>12}  {status}'
        )

    print(sep)
    print(f'  {"TOTAL":<10}  {"":>6}  {"":>5}  {"":>5}  {total_bars:>6}')
    print(f'{"=" * 78}\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description='Forex Bot Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python pipeline.py --full       Complete initial build\n'
            '  python pipeline.py --update     Weekly incremental update\n'
            '  python pipeline.py --validate   Run validation report only\n'
            '  python pipeline.py --status     Print DB status table\n'
        ),
    )
    parser.add_argument('--full',     action='store_true', help='Run complete initial pipeline')
    parser.add_argument('--update',   action='store_true', help='Run weekly incremental update')
    parser.add_argument('--validate', action='store_true', help='Run validation report only')
    parser.add_argument('--status',   action='store_true', help='Print DB status table')
    args = parser.parse_args()

    if   args.full:     run_full_pipeline()
    elif args.update:   run_weekly_update()
    elif args.validate: run_validation()
    elif args.status:   print_status()
    else:               parser.print_help()
