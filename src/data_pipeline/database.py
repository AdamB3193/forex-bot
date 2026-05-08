import os
import sqlite3
from datetime import datetime, timezone

from config import (
    DB_PATH, PAIRS, SPREADS,
    TRAIN_END_DATE, VAL_END_DATE, TEST_START_DATE,
)

_DDL = """
CREATE TABLE IF NOT EXISTS raw_ohlcv (
    pair            TEXT NOT NULL,
    date            TEXT NOT NULL,
    open            REAL NOT NULL,
    high            REAL NOT NULL,
    low             REAL NOT NULL,
    close           REAL NOT NULL,
    tick_volume     REAL NOT NULL DEFAULT 0,
    tick_vol_zscore REAL,
    atr_14          REAL,
    atr_200         REAL,
    is_gap          INTEGER NOT NULL DEFAULT 0,
    is_anomaly      INTEGER NOT NULL DEFAULT 0,
    is_sunday_stub  INTEGER NOT NULL DEFAULT 0,
    spread_est      REAL,
    split_set       TEXT,
    PRIMARY KEY (pair, date)
);

CREATE TABLE IF NOT EXISTS zones (
    zone_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pair               TEXT NOT NULL,
    start_date         TEXT NOT NULL,
    end_date           TEXT,
    top                REAL NOT NULL,
    btm                REAL NOT NULL,
    base_price         REAL NOT NULL,
    is_support         INTEGER NOT NULL,
    is_mitigated       INTEGER NOT NULL DEFAULT 0,
    entries            INTEGER NOT NULL DEFAULT 0,
    sweeps             INTEGER NOT NULL DEFAULT 0,
    strength           INTEGER NOT NULL DEFAULT 0,
    tick_vol_total     REAL NOT NULL DEFAULT 0,
    bounce_quality_avg REAL,
    grade              TEXT,
    is_break_retest    INTEGER NOT NULL DEFAULT 0,
    params_hash        TEXT,
    split_set          TEXT
);

CREATE TABLE IF NOT EXISTS pair_metadata (
    pair               TEXT PRIMARY KEY,
    pip_scale          INTEGER NOT NULL,
    avg_spread         REAL NOT NULL,
    correlation_family TEXT NOT NULL,
    tier               TEXT NOT NULL,
    dukascopy_symbol   TEXT NOT NULL,
    total_bars         INTEGER DEFAULT 0,
    date_from          TEXT,
    date_to            TEXT,
    last_updated       TEXT
);

CREATE TABLE IF NOT EXISTS data_versions (
    version_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    downloaded_at TEXT NOT NULL,
    source        TEXT NOT NULL,
    pair          TEXT NOT NULL,
    date_from     TEXT NOT NULL,
    date_to       TEXT NOT NULL,
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    checksum      TEXT
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_split  ON raw_ohlcv(pair, split_set);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_date   ON raw_ohlcv(pair, date);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_anomaly     ON raw_ohlcv(is_anomaly);",
    "CREATE INDEX IF NOT EXISTS idx_zones_pair_mitig  ON zones(pair, is_mitigated);",
]

_UPSERT_OHLCV = """
INSERT INTO raw_ohlcv (
    pair, date, open, high, low, close, tick_volume,
    tick_vol_zscore, atr_14, atr_200,
    is_gap, is_anomaly, is_sunday_stub, spread_est, split_set
) VALUES (
    :pair, :date, :open, :high, :low, :close, :tick_volume,
    :tick_vol_zscore, :atr_14, :atr_200,
    :is_gap, :is_anomaly, :is_sunday_stub, :spread_est, :split_set
)
ON CONFLICT(pair, date) DO UPDATE SET
    open            = excluded.open,
    high            = excluded.high,
    low             = excluded.low,
    close           = excluded.close,
    tick_volume     = excluded.tick_volume,
    tick_vol_zscore = excluded.tick_vol_zscore,
    atr_14          = excluded.atr_14,
    atr_200         = excluded.atr_200,
    is_gap          = excluded.is_gap,
    is_anomaly      = excluded.is_anomaly,
    is_sunday_stub  = excluded.is_sunday_stub,
    spread_est      = excluded.spread_est,
    split_set       = excluded.split_set;
"""


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _populate_pair_metadata(conn: sqlite3.Connection) -> None:
    rows = [
        (
            symbol,
            meta['pip_scale'],
            SPREADS.get(symbol, 0.0),
            meta['correlation_family'],
            meta['tier'],
            meta['dukascopy_symbol'],
        )
        for symbol, meta in PAIRS.items()
    ]
    conn.executemany(
        """
        INSERT INTO pair_metadata (pair, pip_scale, avg_spread, correlation_family, tier, dukascopy_symbol)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(pair) DO UPDATE SET
            pip_scale          = excluded.pip_scale,
            avg_spread         = excluded.avg_spread,
            correlation_family = excluded.correlation_family,
            tier               = excluded.tier,
            dukascopy_symbol   = excluded.dukascopy_symbol;
        """,
        rows,
    )


def init_database() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.executescript(_DDL)
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)
        _populate_pair_metadata(conn)
        conn.commit()


def _assign_split(date_str: str) -> str:
    if date_str <= TRAIN_END_DATE:
        return 'train'
    if date_str <= VAL_END_DATE:
        return 'val'
    return 'test'


def insert_ohlcv_batch(pair: str, df) -> int:
    """Insert a DataFrame of OHLCV rows for *pair*. Returns rows affected."""
    spread = SPREADS.get(pair, 0.0)
    records = []
    for row in df.itertuples(index=True):
        date_str = str(row.Index) if not isinstance(row.Index, str) else row.Index
        # Allow date column to be either the index or an explicit 'date' column
        if hasattr(row, 'date'):
            date_str = str(row.date)
        records.append({
            'pair':            pair,
            'date':            date_str[:10],
            'open':            float(row.open)        if hasattr(row, 'open')        else float(row.Open),
            'high':            float(row.high)        if hasattr(row, 'high')        else float(row.High),
            'low':             float(row.low)         if hasattr(row, 'low')         else float(row.Low),
            'close':           float(row.close)       if hasattr(row, 'close')       else float(row.Close),
            'tick_volume':     float(row.tick_volume) if hasattr(row, 'tick_volume') else 0.0,
            'tick_vol_zscore': getattr(row, 'tick_vol_zscore', None),
            'atr_14':          getattr(row, 'atr_14',          None),
            'atr_200':         getattr(row, 'atr_200',         None),
            'is_gap':          int(getattr(row, 'is_gap',          0)),
            'is_anomaly':      int(getattr(row, 'is_anomaly',      0)),
            'is_sunday_stub':  int(getattr(row, 'is_sunday_stub',  0)),
            'spread_est':      spread,
            'split_set':       _assign_split(date_str[:10]),
        })

    with get_connection() as conn:
        conn.executemany(_UPSERT_OHLCV, records)
        # Refresh total_bars and date range in pair_metadata
        conn.execute(
            """
            UPDATE pair_metadata
            SET total_bars   = (SELECT COUNT(*) FROM raw_ohlcv WHERE pair = ?),
                date_from    = (SELECT MIN(date) FROM raw_ohlcv WHERE pair = ?),
                date_to      = (SELECT MAX(date) FROM raw_ohlcv WHERE pair = ?),
                last_updated = ?
            WHERE pair = ?
            """,
            (pair, pair, pair, datetime.now(timezone.utc).isoformat(), pair),
        )
        conn.commit()

    return len(records)


if __name__ == '__main__':
    init_database()
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM pair_metadata").fetchone()[0]
    print(f"Database initialized successfully. pair_metadata rows: {count}")
