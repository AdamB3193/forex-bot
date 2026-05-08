"""
Pytest tests for the forex bot data pipeline.

All tests read from the live database at config.DB_PATH.
No TA library imports are used.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data_pipeline'))

from config import (
    DB_PATH,
    KNOWN_ANOMALY_EVENTS,
    PAIRS,
    TEST_START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from database import get_connection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def db():
    """Module-scoped read-only connection to the live database."""
    conn = get_connection()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 1. Database schema
# ---------------------------------------------------------------------------

def test_database_schema(db):
    """All 4 tables exist with the required columns."""
    expected = {
        'raw_ohlcv': {
            'pair', 'date', 'open', 'high', 'low', 'close',
            'tick_volume', 'tick_vol_zscore', 'atr_14', 'atr_200',
            'is_gap', 'is_anomaly', 'is_sunday_stub', 'spread_est', 'split_set',
        },
        'zones': {
            'zone_id', 'pair', 'start_date', 'end_date', 'top', 'btm',
            'base_price', 'is_support', 'is_mitigated', 'entries', 'sweeps',
            'strength', 'tick_vol_total', 'bounce_quality_avg', 'grade',
            'is_break_retest', 'params_hash', 'split_set',
        },
        'pair_metadata': {
            'pair', 'pip_scale', 'avg_spread', 'correlation_family',
            'tier', 'dukascopy_symbol', 'total_bars', 'date_from',
            'date_to', 'last_updated',
        },
        'data_versions': {
            'version_id', 'downloaded_at', 'source', 'pair',
            'date_from', 'date_to', 'rows_inserted', 'checksum',
        },
    }

    for table, required_cols in expected.items():
        exists = db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()[0]
        assert exists == 1, f'Table {table!r} not found in database'

        actual_cols = {
            row[1]
            for row in db.execute(f'PRAGMA table_info({table})').fetchall()
        }
        missing = required_cols - actual_cols
        assert not missing, f'Table {table!r} is missing columns: {missing}'


# ---------------------------------------------------------------------------
# 2. Pair metadata complete
# ---------------------------------------------------------------------------

def test_pair_metadata_complete(db):
    """All 24 pairs have entries in pair_metadata."""
    rows    = db.execute('SELECT pair FROM pair_metadata').fetchall()
    db_pairs = {r[0] for r in rows}
    expected = set(PAIRS.keys())

    missing = expected - db_pairs
    assert not missing, f'Pairs missing from pair_metadata: {missing}'
    assert len(db_pairs) == 24, f'Expected 24 pairs in pair_metadata, got {len(db_pairs)}'


# ---------------------------------------------------------------------------
# 3. No Sunday dates
# ---------------------------------------------------------------------------

def test_no_sunday_dates(db):
    """No row in raw_ohlcv has a Sunday date (strftime '%w' == '0')."""
    count = db.execute(
        "SELECT COUNT(*) FROM raw_ohlcv WHERE strftime('%w', date) = '0'"
    ).fetchone()[0]
    assert count == 0, f'{count} Sunday dates found in raw_ohlcv'


# ---------------------------------------------------------------------------
# 4. No weekend dates
# ---------------------------------------------------------------------------

def test_no_weekend_dates(db):
    """No Saturday or Sunday dates exist anywhere in raw_ohlcv."""
    count = db.execute(
        "SELECT COUNT(*) FROM raw_ohlcv WHERE strftime('%w', date) IN ('0', '6')"
    ).fetchone()[0]
    assert count == 0, f'{count} weekend dates (Sat/Sun) found in raw_ohlcv'


# ---------------------------------------------------------------------------
# 5. OHLCV logic
# ---------------------------------------------------------------------------

def test_ohlcv_logic(db):
    """
    Sample 1000 random non-anomaly, non-gap rows and verify:
        high >= low
        high >= open  and  high >= close
        low  <= open  and  low  <= close
    """
    rows = db.execute(
        """
        SELECT open, high, low, close
        FROM   raw_ohlcv
        WHERE  is_anomaly = 0 AND is_gap = 0
        ORDER  BY RANDOM()
        LIMIT  1000
        """
    ).fetchall()

    assert len(rows) >= 100, f'Too few rows to sample: {len(rows)}'

    violations = []
    for i, (o, h, l, c) in enumerate(rows):
        if h < l:
            violations.append(f'row {i}: high({h}) < low({l})')
        if h < o:
            violations.append(f'row {i}: high({h}) < open({o})')
        if h < c:
            violations.append(f'row {i}: high({h}) < close({c})')
        if l > o:
            violations.append(f'row {i}: low({l}) > open({o})')
        if l > c:
            violations.append(f'row {i}: low({l}) > close({c})')

    assert not violations, (
        f'{len(violations)} OHLCV violation(s):\n' + '\n'.join(violations[:10])
    )


# ---------------------------------------------------------------------------
# 6. Split contamination
# ---------------------------------------------------------------------------

def test_split_no_contamination(db):
    """
    Train rows must not leak past TRAIN_END_DATE.
    Test rows must not appear before TEST_START_DATE.
    Val rows must sit strictly inside the val window.
    """
    # Train rows after TRAIN_END_DATE
    n = db.execute(
        "SELECT COUNT(*) FROM raw_ohlcv WHERE split_set = 'train' AND date > ?",
        (TRAIN_END_DATE,),
    ).fetchone()[0]
    assert n == 0, f'{n} train rows have dates after TRAIN_END_DATE ({TRAIN_END_DATE})'

    # Test rows before TEST_START_DATE
    n = db.execute(
        "SELECT COUNT(*) FROM raw_ohlcv WHERE split_set = 'test' AND date < ?",
        (TEST_START_DATE,),
    ).fetchone()[0]
    assert n == 0, f'{n} test rows have dates before TEST_START_DATE ({TEST_START_DATE})'

    # Val rows outside the val window
    n = db.execute(
        """
        SELECT COUNT(*) FROM raw_ohlcv
        WHERE  split_set = 'val'
          AND  (date <= ? OR date > ?)
        """,
        (TRAIN_END_DATE, VAL_END_DATE),
    ).fetchone()[0]
    assert n == 0, (
        f'{n} val rows outside the val window ({TRAIN_END_DATE} < date <= {VAL_END_DATE})'
    )


# ---------------------------------------------------------------------------
# 7. Known anomalies flagged
# ---------------------------------------------------------------------------

def test_known_anomalies_flagged(db):
    """
    Each entry in KNOWN_ANOMALY_EVENTS that belongs to our 24-pair universe
    must be marked is_anomaly=1 in raw_ohlcv.
    """
    failures = []
    for pair, event_date in KNOWN_ANOMALY_EVENTS:
        if pair not in PAIRS:
            continue  # GBPCHF is not in our universe — skip silently

        row = db.execute(
            'SELECT is_anomaly FROM raw_ohlcv WHERE pair = ? AND date = ?',
            (pair, event_date),
        ).fetchone()

        if row is None:
            failures.append(f'{pair} {event_date}: row not found in DB')
        elif row[0] != 1:
            failures.append(f'{pair} {event_date}: is_anomaly={row[0]}, expected 1')

    assert not failures, (
        'Known anomaly events not correctly flagged:\n' + '\n'.join(failures)
    )


# ---------------------------------------------------------------------------
# 8. ATR not NULL after warmup
# ---------------------------------------------------------------------------

def test_atr_not_null_after_warmup(db):
    """atr_14 must not be NULL for any bar after the 20th row per pair."""
    count = db.execute(
        """
        WITH ranked AS (
            SELECT pair, date, atr_14,
                   ROW_NUMBER() OVER (PARTITION BY pair ORDER BY date) AS rn
            FROM   raw_ohlcv
        )
        SELECT COUNT(*)
        FROM   ranked
        WHERE  rn > 20
          AND  atr_14 IS NULL
        """
    ).fetchone()[0]
    assert count == 0, f'{count} bars past warmup (row > 20) have NULL atr_14'


# ---------------------------------------------------------------------------
# 9. Gap bars have zero volume
# ---------------------------------------------------------------------------

def test_gap_bars_have_zero_volume(db):
    """All synthetic gap bars (is_gap=1) must have tick_volume=0."""
    count = db.execute(
        'SELECT COUNT(*) FROM raw_ohlcv WHERE is_gap = 1 AND tick_volume != 0'
    ).fetchone()[0]
    assert count == 0, f'{count} gap bars have non-zero tick_volume'


# ---------------------------------------------------------------------------
# 10. Pip scale matches ATR order of magnitude
# ---------------------------------------------------------------------------

def test_pip_scale_matches_prices(db):
    """
    The average atr_14 (non-gap bars) for each pair must be in the expected
    price-unit range, confirming the pip scale is correctly applied:

        JPY pairs   (pip_scale=100)   : avg atr_14 in [0.10, 5.00]
        Non-JPY     (pip_scale=10000) : avg atr_14 in [0.001, 0.030]

    Min/max are intentionally not checked here — a 21-year dataset that spans
    GFC 2008, SNB 2015, and COVID 2020 will produce extreme single-day ATR
    readings; the average is the correct signal for scale validation.
    """
    rows = db.execute(
        """
        SELECT pair, AVG(atr_14) AS avg_atr
        FROM   raw_ohlcv
        WHERE  is_gap = 0 AND atr_14 IS NOT NULL
        GROUP  BY pair
        """
    ).fetchall()

    assert rows, 'No ATR data found in database'

    failures = []
    for pair, avg_atr in rows:
        if pair not in PAIRS:
            continue
        pip_scale = PAIRS[pair]['pip_scale']
        is_jpy    = (pip_scale == 100)
        lo, hi    = (0.10, 5.00) if is_jpy else (0.001, 0.030)

        if avg_atr is None or not (lo <= avg_atr <= hi):
            failures.append(
                f'{pair}: avg_atr={avg_atr:.6f} not in [{lo}, {hi}] '
                f'(pip_scale={pip_scale})'
            )

    assert not failures, 'ATR pip-scale mismatches:\n' + '\n'.join(failures)
