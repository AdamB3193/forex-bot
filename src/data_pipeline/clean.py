"""
Clean and enrich raw OHLCV data for all 24 pairs.

Pipeline (per pair):
    1.  normalize_pair_timezone  – remove Sunday stubs, align to NY close
    2.  compute_atr(14)          – Wilder ATR-14 (manual EWM, no TA library)
    3.  compute_atr(200)         – ATR-200 for long-range volatility context
    4.  detect_anomalies         – flag flash crashes; cap ranges
    5.  fill_gaps                – forward-fill missing business days
    6.  normalize_tick_volume    – z-score; gap bars get -3.0
    7.  assign_split_set         – train / val / test labels
    8.  assign_spread            – spread_est from config

ATR NOTE: All ATR calculations are done manually via numpy/pandas EWM.
Do NOT use ta.volatility.AverageTrueRange or pandas_ta anywhere in this file.
"""

import bisect
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    ANOMALY_ATR_MULTIPLE,
    KNOWN_ANOMALY_EVENTS,
    PAIRS,
    SPREADS,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from database import get_connection, insert_ohlcv_batch
from timezone_fix import normalize_pair_timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ATR — manual Wilder EWM (no TA library)
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's Average True Range computed entirely in numpy.

    True Range  = max(H−L, |H−PrevC|, |L−PrevC|)
    Warmup      : first *period* bars use expanding simple mean
    Main        : EWM with alpha = 1/period, seeded by warmup mean
    """
    hi  = df['high'].to_numpy(dtype=float)
    lo  = df['low'].to_numpy(dtype=float)
    cl  = df['close'].to_numpy(dtype=float)

    prev_cl      = np.empty_like(cl)
    prev_cl[0]   = cl[0]
    prev_cl[1:]  = cl[:-1]

    tr = np.maximum(hi - lo,
         np.maximum(np.abs(hi - prev_cl),
                    np.abs(lo  - prev_cl)))

    n     = len(tr)
    alpha = 1.0 / period
    atr   = np.empty(n, dtype=float)

    # Warmup: expanding simple mean for indices 0 … period-1
    for i in range(min(period, n)):
        atr[i] = tr[: i + 1].mean()

    # Wilder EWM seeded by the simple mean of the warmup window
    if n > period:
        prev = tr[:period].mean()
        for i in range(period, n):
            prev   = prev * (1.0 - alpha) + tr[i] * alpha
            atr[i] = prev

    return pd.Series(atr, index=df.index)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag anomalous bars and cap their high/low range.

    Priority order:
        1. KNOWN_ANOMALY_EVENTS from config  (hard-coded flash crashes)
        2. Range > ANOMALY_ATR_MULTIPLE × ATR-14
        3. Body  > 3 × ATR-14  (open-to-close move)

    All flagged bars have their high/low replaced with
        close ± ANOMALY_ATR_MULTIPLE × atr_14
    so downstream zone detection sees a bounded range.
    """
    df = df.copy()
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = 0

    # 1. Hard-coded known events
    known_dates = {date for p, date in KNOWN_ANOMALY_EVENTS if p == pair}
    if known_dates:
        df.loc[df['date'].isin(known_dates), 'is_anomaly'] = 1

    # 2. ATR-based range filter
    cap = ANOMALY_ATR_MULTIPLE * df['atr_14']
    df.loc[(df['high'] - df['low']) > cap, 'is_anomaly'] = 1

    # 3. Large body filter
    df.loc[(df['close'] - df['open']).abs() > 3.0 * df['atr_14'], 'is_anomaly'] = 1

    # 4. Cap ranges for all flagged bars
    mask = df['is_anomaly'] == 1
    if mask.any():
        cap_vals              = ANOMALY_ATR_MULTIPLE * df.loc[mask, 'atr_14']
        df.loc[mask, 'high']  = df.loc[mask, 'close'] + cap_vals
        df.loc[mask, 'low']   = df.loc[mask, 'close'] - cap_vals
        log.info('[%s] %d anomaly bars flagged and range-capped', pair, mask.sum())

    return df


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def _forex_holidays(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """
    New Year's Day (Jan 1) and Christmas Day (Dec 25), adjusted to nearest
    weekday, for every year in [start_year, end_year].
    """
    dates = []
    for year in range(start_year, end_year + 1):
        for month, day in ((1, 1), (12, 25)):
            h = pd.Timestamp(year, month, day)
            if   h.dayofweek == 5:   # Saturday -> Friday
                h -= pd.Timedelta(days=1)
            elif h.dayofweek == 6:   # Sunday -> Monday
                h += pd.Timedelta(days=1)
            dates.append(h)
    return pd.DatetimeIndex(dates)


def fill_gaps(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert a forward-filled bar for every business day missing from *df*.

    Excludes New Year's Day and Christmas from the expected sequence so that
    genuine market closures are not treated as gaps.
    """
    start = df['date'].min()
    end   = df['date'].max()

    holidays = _forex_holidays(
        pd.Timestamp(start).year,
        pd.Timestamp(end).year,
    )
    from pandas.tseries.offsets import CustomBusinessDay
    cbd = CustomBusinessDay(holidays=list(holidays))
    expected_strs = set(
        pd.date_range(start=start, end=end, freq=cbd).strftime('%Y-%m-%d')
    )

    existing = set(df['date'])
    missing  = sorted(expected_strs - existing)

    if not missing:
        return df

    # Build a sorted-date -> close lookup for O(log n) prev-close lookup
    sorted_dates = sorted(df['date'].tolist())
    date_close   = dict(zip(df['date'], df['close']))

    gap_rows = []
    for gap_date in missing:
        idx = bisect.bisect_left(sorted_dates, gap_date) - 1
        if idx < 0:
            continue
        prev_close = date_close[sorted_dates[idx]]
        gap_rows.append({
            'date':          gap_date,
            'open':          prev_close,
            'high':          prev_close,
            'low':           prev_close,
            'close':         prev_close,
            'tick_volume':   0.0,
            'is_gap':        1,
            'is_anomaly':    0,
            'is_sunday_stub': 0,
        })

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        df     = pd.concat([df, gap_df], ignore_index=True)
        df     = df.sort_values('date').reset_index(drop=True)

        # ATR was computed before gaps were inserted — forward-fill for gap rows
        for col in ('atr_14', 'atr_200'):
            if col in df.columns:
                df[col] = df[col].ffill()

        # Ensure integer flag columns have no NaN (concat introduces NaN for
        # columns absent from gap_df)
        for col in ('is_gap', 'is_anomaly', 'is_sunday_stub'):
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        log.info('[%s] %d gap bars forward-filled', pair, len(gap_rows))

    return df


# ---------------------------------------------------------------------------
# Tick-volume z-score
# ---------------------------------------------------------------------------

def normalize_tick_volume(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise tick_volume using statistics from non-gap bars only.
    Gap bars (is_gap == 1) receive a fixed z-score of -3.0.
    """
    df       = df.copy()
    non_gap  = (df['is_gap'] == 0) if 'is_gap' in df.columns else pd.Series(True, index=df.index)
    vols     = df.loc[non_gap, 'tick_volume']
    mu, sigma = vols.mean(), vols.std()

    if sigma > 0:
        df['tick_vol_zscore'] = (df['tick_volume'] - mu) / sigma
    else:
        df['tick_vol_zscore'] = 0.0

    if 'is_gap' in df.columns:
        df.loc[df['is_gap'] == 1, 'tick_vol_zscore'] = -3.0

    return df


# ---------------------------------------------------------------------------
# Split assignment and spread
# ---------------------------------------------------------------------------

def assign_split_set(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row as 'train', 'val', or 'test' based on the date."""
    df = df.copy()
    df['split_set'] = np.select(
        [df['date'] <= TRAIN_END_DATE,
         (df['date'] > TRAIN_END_DATE) & (df['date'] <= VAL_END_DATE)],
        ['train', 'val'],
        default='test',
    )
    return df


def assign_spread(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['spread_est'] = SPREADS.get(pair, 0.0)
    return df


# ---------------------------------------------------------------------------
# Per-pair orchestrator
# ---------------------------------------------------------------------------

def clean_pair(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline for one pair.

    Input  : raw DataFrame from the DB (date, open, high, low, close, tick_volume, flags)
    Output : fully enriched DataFrame ready for upsert
    """
    # Ensure flag columns exist and are integer-typed
    for col, default in (('is_gap', 0), ('is_anomaly', 0), ('is_sunday_stub', 0)):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(0).astype(int)

    # Real bars (tick_volume > 0) must never carry a stale is_gap=1 flag.
    # fill_gaps is the sole authority that sets is_gap=1; any row with real
    # volume came from Dukascopy and is not a synthetic gap.
    df.loc[df['tick_volume'] > 0, 'is_gap'] = 0

    # Step 1 — timezone / Sunday-stub normalisation
    df = normalize_pair_timezone(pair, df)
    df = df.sort_values('date').reset_index(drop=True)

    # Step 2 — ATR-14 and ATR-200 (manual Wilder EWM)
    df['atr_14']  = compute_atr(df, 14)
    df['atr_200'] = compute_atr(df, 200)

    # Step 3 — anomaly detection + range cap
    df = detect_anomalies(pair, df)

    # Step 4 — gap filling (forward-fills ATR for inserted rows)
    df = fill_gaps(pair, df)

    # Step 5 — tick-volume z-score
    df = normalize_tick_volume(pair, df)

    # Step 6 — train / val / test split
    df = assign_split_set(df)

    # Step 7 — estimated spread
    df = assign_spread(pair, df)

    return df


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------

def _load_raw_from_db(pair: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, tick_volume,
                   COALESCE(is_gap,         0) AS is_gap,
                   COALESCE(is_anomaly,     0) AS is_anomaly,
                   COALESCE(is_sunday_stub, 0) AS is_sunday_stub
            FROM   raw_ohlcv
            WHERE  pair = ?
            ORDER  BY date
            """,
            conn,
            params=(pair,),
        )
    return df


# ---------------------------------------------------------------------------
# Batch cleaner
# ---------------------------------------------------------------------------

def clean_all_pairs() -> None:
    print(f'\n{"=" * 78}')
    print(f'Cleaning {len(PAIRS)} pairs')
    print(f'{"=" * 78}\n')

    results = {}

    for pair in PAIRS:
        try:
            raw = _load_raw_from_db(pair)
            if raw.empty:
                print(f'[{pair:10s}] SKIP — no data in DB')
                results[pair] = 'SKIP'
                continue

            raw_rows = len(raw)
            cleaned  = clean_pair(pair, raw)

            rows_upserted = insert_ohlcv_batch(pair, cleaned.set_index('date'))

            # Remove weekend rows that were merged/dropped in the DataFrame but
            # still exist in the DB from the original download (upsert doesn't delete)
            with get_connection() as conn:
                deleted = conn.execute(
                    "DELETE FROM raw_ohlcv WHERE pair = ? "
                    "AND strftime('%w', date) IN ('0', '6')",
                    (pair,),
                ).rowcount
                conn.commit()
            if deleted:
                log.info('[%s] Deleted %d weekend rows from DB', pair, deleted)

            # Refresh pair_metadata now that weekend rows have been deleted
            with get_connection() as conn:
                conn.execute(
                    """
                    UPDATE pair_metadata
                    SET total_bars   = (SELECT COUNT(*)  FROM raw_ohlcv WHERE pair = ?),
                        date_from    = (SELECT MIN(date) FROM raw_ohlcv WHERE pair = ?),
                        date_to      = (SELECT MAX(date) FROM raw_ohlcv WHERE pair = ?),
                        last_updated = ?
                    WHERE pair = ?
                    """,
                    (pair, pair, pair, datetime.now(timezone.utc).isoformat(), pair),
                )
                conn.commit()

            n_gaps    = int(cleaned['is_gap'].sum())
            n_anomaly = int(cleaned['is_anomaly'].sum())
            n_train   = int((cleaned['split_set'] == 'train').sum())
            n_val     = int((cleaned['split_set'] == 'val').sum())
            n_test    = int((cleaned['split_set'] == 'test').sum())

            print(
                f'[{pair:10s}]  raw={raw_rows:>5} -> clean={rows_upserted:>5}  '
                f'gaps={n_gaps:>3}  anomalies={n_anomaly:>2}  '
                f'train={n_train:>4}  val={n_val:>4}  test={n_test:>4}'
            )
            results[pair] = 'OK'

        except Exception as exc:
            import traceback
            print(f'[{pair:10s}] ERROR: {exc}')
            traceback.print_exc()
            results[pair] = f'ERROR: {exc}'

    ok   = sum(1 for v in results.values() if v == 'OK')
    skip = sum(1 for v in results.values() if v == 'SKIP')
    fail = len(results) - ok - skip

    print(f'\n{"=" * 78}')
    print(f'Completed: {ok} OK | {skip} skipped | {fail} failed')
    print(f'{"=" * 78}\n')


if __name__ == '__main__':
    clean_all_pairs()
