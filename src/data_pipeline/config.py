import os

PAIRS = {
    'EURUSD': {'pip_scale': 10000, 'tier': 'A', 'correlation_family': 'USD_weakness', 'dukascopy_symbol': 'EURUSD'},
    'GBPUSD': {'pip_scale': 10000, 'tier': 'A', 'correlation_family': 'USD_weakness', 'dukascopy_symbol': 'GBPUSD'},
    'USDJPY': {'pip_scale': 100,   'tier': 'A', 'correlation_family': 'JPY',          'dukascopy_symbol': 'USDJPY'},
    'USDCHF': {'pip_scale': 10000, 'tier': 'A', 'correlation_family': 'USD_strength', 'dukascopy_symbol': 'USDCHF'},
    'AUDUSD': {'pip_scale': 10000, 'tier': 'A', 'correlation_family': 'USD_weakness', 'dukascopy_symbol': 'AUDUSD'},
    'EURGBP': {'pip_scale': 10000, 'tier': 'A', 'correlation_family': 'EUR',          'dukascopy_symbol': 'EURGBP'},
    'EURJPY': {'pip_scale': 100,   'tier': 'B', 'correlation_family': 'JPY',          'dukascopy_symbol': 'EURJPY'},
    'GBPJPY': {'pip_scale': 100,   'tier': 'B', 'correlation_family': 'GBP_JPY',      'dukascopy_symbol': 'GBPJPY'},
    'EURAUD': {'pip_scale': 10000, 'tier': 'B', 'correlation_family': 'EUR',          'dukascopy_symbol': 'EURAUD'},
    'AUDJPY': {'pip_scale': 100,   'tier': 'B', 'correlation_family': 'JPY',          'dukascopy_symbol': 'AUDJPY'},
    'USDCAD': {'pip_scale': 10000, 'tier': 'B', 'correlation_family': 'USD_strength', 'dukascopy_symbol': 'USDCAD'},
    'EURCAD': {'pip_scale': 10000, 'tier': 'B', 'correlation_family': 'EUR',          'dukascopy_symbol': 'EURCAD'},
    'CADJPY': {'pip_scale': 100,   'tier': 'B', 'correlation_family': 'JPY',          'dukascopy_symbol': 'CADJPY'},
    'GBPAUD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'GBP',          'dukascopy_symbol': 'GBPAUD'},
    'GBPCAD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'GBP',          'dukascopy_symbol': 'GBPCAD'},
    'GBPNZD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'GBP',          'dukascopy_symbol': 'GBPNZD'},
    'EURNZD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'EUR',          'dukascopy_symbol': 'EURNZD'},
    'AUDCAD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'AUD',          'dukascopy_symbol': 'AUDCAD'},
    'NZDUSD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'USD_weakness', 'dukascopy_symbol': 'NZDUSD'},
    'CHFJPY': {'pip_scale': 100,   'tier': 'C', 'correlation_family': 'JPY',          'dukascopy_symbol': 'CHFJPY'},
    'NZDJPY': {'pip_scale': 100,   'tier': 'C', 'correlation_family': 'JPY',          'dukascopy_symbol': 'NZDJPY'},
    'AUDNZD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'AUD',          'dukascopy_symbol': 'AUDNZD'},
    'EURCHF': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'EUR',          'dukascopy_symbol': 'EURCHF'},
    'NZDCAD': {'pip_scale': 10000, 'tier': 'C', 'correlation_family': 'AUD',          'dukascopy_symbol': 'NZDCAD'},
}

# Data date ranges - LOCKED. Do not change after database is created.
DATA_START_DATE = '2005-01-01'
DATA_END_DATE   = '2026-04-30'

# Train/validation/test split - LOCKED. Never touch test set during development.
TRAIN_END_DATE  = '2019-12-31'
VAL_END_DATE    = '2022-12-31'
TEST_START_DATE = '2023-01-01'   # Test set begins here. NEVER used during training.

# Timezone: all daily candles close at NY close (17:00 EST = 22:00 UTC winter, 21:00 UTC summer)
TARGET_TZ = 'America/New_York'
CANDLE_CLOSE_HOUR_NY = 17  # 5 PM New York time

# Anomaly detection threshold: flag candles where range > this multiple of ATR_14
ANOMALY_ATR_MULTIPLE = 5.0

# Estimated average spreads per pair (in pips) for backtest cost simulation
SPREADS = {
    'EURUSD': 0.5, 'GBPUSD': 0.8, 'USDJPY': 0.8, 'USDCHF': 1.0,
    'AUDUSD': 0.8, 'EURGBP': 0.8, 'EURJPY': 1.2, 'GBPJPY': 2.0,
    'EURAUD': 1.5, 'AUDJPY': 1.5, 'USDCAD': 1.2, 'EURCAD': 1.5,
    'CADJPY': 1.8, 'GBPAUD': 3.0, 'GBPCAD': 3.0, 'GBPNZD': 5.0,
    'EURNZD': 3.5, 'AUDCAD': 2.0, 'NZDUSD': 1.5, 'CHFJPY': 2.0,
    'NZDJPY': 2.0, 'AUDNZD': 2.5, 'EURCHF': 1.5, 'NZDCAD': 2.5,
}

# Database and data paths - all relative to project root
_ROOT = r'C:\Users\Adam\OneDrive\Documents\Projects\Forex Bot'
DB_PATH            = os.path.join(_ROOT, 'data', 'db', 'forex_bot.db')
RAW_DATA_DIR       = os.path.join(_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(_ROOT, 'data', 'processed')
LOGS_DIR           = os.path.join(_ROOT, 'logs')

# Known flash crash / extreme event dates to flag as anomalies regardless of ATR filter
KNOWN_ANOMALY_EVENTS = [
    ('EURCHF', '2015-01-15'),  # SNB peg removal - 40% crash in seconds
    ('USDCHF', '2015-01-15'),
    ('CHFJPY', '2015-01-15'),
    ('GBPCHF', '2015-01-15'),
    ('GBPUSD', '2016-10-07'),  # GBP flash crash - 6% in 2 minutes
    ('GBPJPY', '2016-10-07'),
    ('GBPAUD', '2016-10-07'),
    ('GBPCAD', '2016-10-07'),
    ('GBPNZD', '2016-10-07'),
    ('USDJPY', '2019-01-03'),  # JPY flash crash
    ('AUDUSD', '2019-01-03'),
    ('AUDJPY', '2019-01-03'),
]

# Minimum bars required per pair for it to be considered usable
MIN_BARS_REQUIRED = 3000  # roughly 12 years of trading days

# TA LIBRARY NOTE: This project uses the 'ta' library (import ta) for any technical
# indicator helpers. Do NOT use pandas_ta anywhere in this codebase — it does not
# support Python 3.11. All ATR calculations are done manually using pandas EWM
# (see clean.py). The 'ta' library is available as a fallback for any other
# indicator needs that arise.
