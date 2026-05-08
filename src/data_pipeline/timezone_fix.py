"""
Normalise Dukascopy daily candles to NY close (17:00 EST) day boundaries.

Dukascopy labels each bar by its UTC-midnight open timestamp, which produces
two kinds of weekend stub candles:

  Sunday stub  (22:00-00:00 UTC): the first 2 hours of the trading week.
               Merged into Monday so Monday's open reflects the week-open price.

  Saturday stub (00:00-02:00 UTC Saturday, i.e. 22:00-00:00 UTC Friday):
               2 hours after NY close on Friday. In NY-close convention these
               fall outside any daily candle and are simply dropped.

After stub removal the UTC-midnight date label equals the correct NY-close
label for the trading session that ended on that calendar date.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

_SUNDAY   = 6   # pd.Timestamp.dayofweek
_SATURDAY = 5


# ---------------------------------------------------------------------------
# Convention detection
# ---------------------------------------------------------------------------

def detect_candle_close_convention(df: pd.DataFrame) -> str:
    """
    Return 'UTC_MIDNIGHT', 'NY_CLOSE', or 'UNKNOWN' based on the date sequence.
    """
    if df.empty or 'date' not in df.columns:
        return 'UNKNOWN'
    dates = pd.to_datetime(df['date'], errors='coerce').dropna()
    if len(dates) < 10:
        return 'UNKNOWN'
    has_weekend = ((dates.dt.dayofweek == _SUNDAY) | (dates.dt.dayofweek == _SATURDAY)).any()
    return 'UTC_MIDNIGHT' if has_weekend else 'NY_CLOSE'


# ---------------------------------------------------------------------------
# Weekend stub removal
# ---------------------------------------------------------------------------

def remove_sunday_stubs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Sunday partial bars into the following Monday; drop Saturday stubs.

    Sunday: fold open/high/low/volume into Monday.
    Saturday: drop without merging (after-NY-close stub, no canonical candle).
    """
    df       = df.copy()
    dates_ts = pd.to_datetime(df['date'])

    sunday_dates   = df.loc[dates_ts.dt.dayofweek == _SUNDAY,   'date'].tolist()
    saturday_dates = df.loc[dates_ts.dt.dayofweek == _SATURDAY, 'date'].tolist()

    if not sunday_dates and not saturday_dates:
        return df

    df = df.set_index('date')

    # Sunday: merge into Monday
    for sun_date in sunday_dates:
        if sun_date not in df.index:
            continue
        mon_date = (pd.Timestamp(sun_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        sun      = df.loc[sun_date]
        if mon_date in df.index:
            df.at[mon_date, 'open']        = float(sun['open'])
            df.at[mon_date, 'high']        = max(float(sun['high']), float(df.at[mon_date, 'high']))
            df.at[mon_date, 'low']         = min(float(sun['low']),  float(df.at[mon_date, 'low']))
            df.at[mon_date, 'tick_volume'] = (float(sun.get('tick_volume', 0)) +
                                              float(df.at[mon_date, 'tick_volume']))
        df.drop(index=sun_date, inplace=True)

    # Saturday: drop without merging
    for sat_date in saturday_dates:
        if sat_date in df.index:
            df.drop(index=sat_date, inplace=True)

    df = df.reset_index()
    df = df.sort_values('date').reset_index(drop=True)
    log.info('Removed %d Sunday + %d Saturday stubs',
             len(sunday_dates), len(saturday_dates))
    return df


# ---------------------------------------------------------------------------
# Top-level normalisation
# ---------------------------------------------------------------------------

def fix_timezone_to_ny_close(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply UTC-midnight -> NY-close alignment for *pair*."""
    convention = detect_candle_close_convention(df)

    if convention == 'NY_CLOSE':
        log.debug('[%s] Already NY-close aligned (%d rows)', pair, len(df))
        return df

    if convention == 'UTC_MIDNIGHT':
        before = len(df)
        df     = remove_sunday_stubs(df)
        log.info('[%s] %d weekend stubs removed -> %d rows remain',
                 pair, before - len(df), len(df))
        return df

    log.warning('[%s] Convention unknown -- skipping timezone fix', pair)
    return df


def normalize_pair_timezone(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """Top-level entry point: return *df* with correct NY-close date labels."""
    return fix_timezone_to_ny_close(pair, df)
