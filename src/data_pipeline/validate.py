"""
Data quality validation report for the forex bot data pipeline.

Runs 8 checks across all 24 pairs and writes the result to:
  - terminal (stdout)
  - logs/validation_report_{TIMESTAMP}.txt

No TA library (ta, pandas_ta) is used or needed here.
"""

import os
import sys
from datetime import datetime, timezone

# Must reconfigure stdout for UTF-8 before any print so check/cross symbols
# render correctly on Windows (default cp1252 terminal).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DB_PATH, LOGS_DIR, PAIRS, SPREADS,
    MIN_BARS_REQUIRED, KNOWN_ANOMALY_EVENTS,
    TRAIN_END_DATE, VAL_END_DATE, TEST_START_DATE,
)
from database import get_connection

OK   = '  ✓ PASS'
FAIL = '  ✗ FAIL'
WARN = '  ! WARN'


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
# Formatting helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print(f'\n{"=" * 72}')
    print(f'  {title}')
    print(f'{"=" * 72}')


def _sub(title: str) -> None:
    print(f'\n  -- {title} --')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_validation() -> None:
    ts          = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(LOGS_DIR, f'validation_report_{ts}.txt')

    tee = _Tee(report_path)
    _original = sys.stdout
    sys.stdout = tee

    try:
        _run_all_checks()
    finally:
        sys.stdout = _original
        tee.close()

    print(f'Report saved -> {report_path}')


# ---------------------------------------------------------------------------
# All checks
# ---------------------------------------------------------------------------

def _run_all_checks() -> None:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'\n{"=" * 72}')
    print(f'  FOREX BOT DATA PIPELINE -- VALIDATION REPORT')
    print(f'  Generated : {now}')
    print(f'  Database  : {DB_PATH}')
    print(f'{"=" * 72}')

    conn = get_connection()

    global_issues     = 0
    global_fail_pairs = 0
    test_contaminated = False

    # =========================================================================
    # CHECK 1 — Per-pair completeness
    # =========================================================================
    _hdr('CHECK 1 -- PER-PAIR DATA COMPLETENESS')

    completeness = conn.execute('''
        SELECT
            pair,
            COUNT(*)                                                           AS total,
            MIN(date)                                                          AS date_from,
            MAX(date)                                                          AS date_to,
            SUM(CASE WHEN split_set = "train" THEN 1 ELSE 0 END)               AS n_train,
            SUM(CASE WHEN split_set = "val"   THEN 1 ELSE 0 END)               AS n_val,
            SUM(CASE WHEN split_set = "test"  THEN 1 ELSE 0 END)               AS n_test,
            SUM(CASE WHEN is_gap     = 1 THEN 1 ELSE 0 END)                    AS n_gaps,
            SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END)                    AS n_anom,
            SUM(CASE WHEN open IS NULL OR high IS NULL
                          OR low  IS NULL OR close IS NULL THEN 1 ELSE 0 END)  AS n_null
        FROM raw_ohlcv
        GROUP BY pair
        ORDER BY pair
    ''').fetchall()

    col_hdr = (f'  {"PAIR":<10}  {"TOTAL":>6}  {"FROM":>12}  {"TO":>12}  '
               f'{"TRAIN":>6}  {"VAL":>5}  {"TEST":>5}  '
               f'{"GAPS%":>6}  {"ANOM%":>6}  {"NULLS":>5}  STATUS')
    print(col_hdr)
    print('  ' + '-' * 100)

    pair_stats     = {}
    total_bars_all = 0

    for r in completeness:
        pair, total, dfrom, dto, n_train, n_val, n_test, n_gaps, n_anom, n_null = r
        total_bars_all += total

        pair_ok = (total >= MIN_BARS_REQUIRED and n_train >= 2000 and n_null == 0)
        sym = OK if pair_ok else FAIL
        if not pair_ok:
            global_issues     += 1
            global_fail_pairs += 1

        g_pct = 100 * n_gaps / total if total else 0
        a_pct = 100 * n_anom / total if total else 0

        print(f'  {pair:<10}  {total:>6}  {dfrom:>12}  {dto:>12}  '
              f'{n_train:>6}  {n_val:>5}  {n_test:>5}  '
              f'{g_pct:5.1f}%  {a_pct:5.1f}%  {n_null:>5}  {sym}')

        pair_stats[pair] = dict(
            total=total, date_from=dfrom, date_to=dto,
            n_train=n_train, n_val=n_val, n_test=n_test,
            n_gaps=n_gaps, n_anom=n_anom, n_null=n_null,
        )

    print(f'\n  Total bars across all 24 pairs : {total_bars_all:,}')
    print(f'  Pairs failing completeness      : {global_fail_pairs}')

    # =========================================================================
    # CHECK 2 — OHLCV logic (sample 100 non-anomaly bars per pair)
    # =========================================================================
    _hdr('CHECK 2 -- OHLCV LOGIC (100 random non-anomaly bars per pair)')

    ohlcv_total_violations = 0
    for pair in PAIRS:
        sample = conn.execute('''
            SELECT open, high, low, close
            FROM raw_ohlcv
            WHERE pair = ? AND is_anomaly = 0
            ORDER BY RANDOM() LIMIT 100
        ''', (pair,)).fetchall()

        v = dict.fromkeys(
            ['high<open', 'high<close', 'low>open', 'low>close',
             'high<low', 'open<=0', 'close<=0'], 0
        )
        for o, h, l, c in sample:
            if h < o:   v['high<open']  += 1
            if h < c:   v['high<close'] += 1
            if l > o:   v['low>open']   += 1
            if l > c:   v['low>close']  += 1
            if h < l:   v['high<low']   += 1
            if o <= 0:  v['open<=0']    += 1
            if c <= 0:  v['close<=0']   += 1

        n_v = sum(v.values())
        if n_v == 0:
            print(f'  {pair:<10}  {OK}  ({len(sample)} bars sampled)')
        else:
            detail = ', '.join(f'{k}:{n}' for k, n in v.items() if n)
            print(f'  {pair:<10}  {FAIL}  {n_v} violation(s): {detail}')
            ohlcv_total_violations += n_v
            global_issues          += n_v

    print(f'\n  Total OHLCV violations : {ohlcv_total_violations}')

    # =========================================================================
    # CHECK 3 — Timezone / weekend consistency
    # =========================================================================
    _hdr('CHECK 3 -- TIMEZONE / WEEKEND CONSISTENCY')

    weekend_counts = {
        p: c for p, c in conn.execute('''
            SELECT pair, COUNT(*) AS cnt
            FROM raw_ohlcv
            WHERE strftime("%w", date) IN ("0","6")
            GROUP BY pair
        ''').fetchall()
    }

    tz_fail = 0
    for pair in PAIRS:
        cnt = weekend_counts.get(pair, 0)
        if cnt == 0:
            print(f'  {pair:<10}  {OK}  no weekend dates')
        else:
            print(f'  {pair:<10}  {FAIL}  {cnt} weekend date(s) found')
            tz_fail       += cnt
            global_issues += cnt

    print(f'\n  Total weekend rows remaining : {tz_fail}')

    # =========================================================================
    # CHECK 4 — Train / val / test split integrity
    # =========================================================================
    _hdr('CHECK 4 -- TRAIN / VAL / TEST SPLIT INTEGRITY')

    split_bounds = conn.execute('''
        SELECT split_set,
               MIN(date) AS min_d,
               MAX(date) AS max_d,
               COUNT(*)  AS n
        FROM raw_ohlcv
        GROUP BY split_set
        ORDER BY min_d
    ''').fetchall()

    print(f'\n  Global split boundaries (all pairs combined):')
    print(f'  {"SPLIT":<8}  {"FROM":>12}  {"TO":>12}  {"BARS":>10}')
    print(f'  {"-"*8}  {"-"*12}  {"-"*12}  {"-"*10}')
    for s, mn, mx, n in split_bounds:
        print(f'  {(s or "NULL"):<8}  {(mn or "--"):>12}  {(mx or "--"):>12}  {n:>10,}')

    # Contamination checks
    test_in_train = conn.execute(
        'SELECT COUNT(*) FROM raw_ohlcv WHERE date >= ? AND split_set = "train"',
        (TEST_START_DATE,)
    ).fetchone()[0]

    train_in_test = conn.execute(
        'SELECT COUNT(*) FROM raw_ohlcv WHERE date <= ? AND split_set = "test"',
        (TRAIN_END_DATE,)
    ).fetchone()[0]

    val_out_of_window = conn.execute(
        'SELECT COUNT(*) FROM raw_ohlcv WHERE split_set = "val" AND (date <= ? OR date > ?)',
        (TRAIN_END_DATE, VAL_END_DATE)
    ).fetchone()[0]

    print()
    for label, count in [
        ('Test dates in train split  ', test_in_train),
        ('Train dates in test split  ', train_in_test),
        ('Val dates outside val window', val_out_of_window),
    ]:
        sym = OK if count == 0 else FAIL
        print(f'  {label}: {count:>6}  {sym}')
        if count > 0:
            global_issues     += count
            test_contaminated  = True

    print()
    if test_contaminated:
        print(f'  ✗ TEST SET INTEGRITY: CONTAMINATED')
        print(f'    !! DO NOT PROCEED -- fix split assignment before training !!')
    else:
        print(f'  ✓ TEST SET INTEGRITY: CLEAN')

    print(f'\n  Date boundaries from config:')
    print(f'    Train : 2005-01-01 .. {TRAIN_END_DATE}')
    val_start_year = str(int(TRAIN_END_DATE[:4]) + 1)
    print(f'    Val   : {(val_start_year + "-01-01"):>12} (next year) .. {VAL_END_DATE}')
    print(f'    Test  : {TEST_START_DATE} .. {pair_stats.get("EURUSD", {}).get("date_to","?")}')

    # =========================================================================
    # CHECK 5 — Known anomaly events confirmed
    # =========================================================================
    _hdr('CHECK 5 -- KNOWN ANOMALY EVENTS CONFIRMED IN DB')

    missing_flags = 0
    for pair, date in KNOWN_ANOMALY_EVENTS:
        if pair not in PAIRS:
            print(f'  {pair:<10}  {date}  {WARN}  pair not in our 24-pair universe (expected)')
            continue

        row = conn.execute(
            'SELECT is_anomaly FROM raw_ohlcv WHERE pair=? AND date=?',
            (pair, date)
        ).fetchone()

        if row is None:
            print(f'  {pair:<10}  {date}  {WARN}  row not found in DB')
        elif row[0] == 1:
            print(f'  {pair:<10}  {date}  {OK}  is_anomaly=1')
        else:
            print(f'  {pair:<10}  {date}  {FAIL}  is_anomaly={row[0]} (expected 1)')
            missing_flags += 1
            global_issues += 1

    status = OK if missing_flags == 0 else FAIL
    print(f'\n  {len(KNOWN_ANOMALY_EVENTS)} events checked | {missing_flags} incorrectly flagged  {status}')

    # =========================================================================
    # CHECK 6 — ATR-14 sanity check
    # =========================================================================
    _hdr('CHECK 6 -- ATR-14 SANITY CHECK')

    atr_data = conn.execute('''
        SELECT pair,
               MIN(atr_14) AS mn,
               MAX(atr_14) AS mx,
               AVG(atr_14) AS avg,
               SUM(CASE WHEN atr_14 IS NULL THEN 1 ELSE 0 END) AS n_null
        FROM raw_ohlcv
        WHERE is_gap = 0
        GROUP BY pair
        ORDER BY pair
    ''').fetchall()

    # Expected avg ATR range in pips — catches wrong pip_scale but not historical extremes.
    # Min/max over 21 years will include GFC/COVID spikes; we only FAIL on NULLs or
    # avg pips outside the plausible scale window.
    _AVG_PIPS_LO = 20.0    # below → likely wrong scale (e.g. price stored as pips)
    _AVG_PIPS_HI = 800.0   # above → likely wrong scale

    atr_map = {}
    atr_issues = 0
    print(f'\n  {"PAIR":<10}  {"AVG(pips)":>10}  {"MIN":>10}  {"MAX":>10}  {"NULL_ATR":>8}  STATUS')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*14}')
    for pair, mn, mx, avg, n_null in atr_data:
        atr_map[pair] = avg
        pip_scale = PAIRS[pair]['pip_scale']
        avg_pips  = (avg * pip_scale) if avg else 0.0

        # FAIL: any NULLs, or average out of plausible pip range (wrong scale)
        if n_null > 0 or avg_pips < _AVG_PIPS_LO or avg_pips > _AVG_PIPS_HI:
            sym = FAIL
            atr_issues    += 1
            global_issues += 1
        else:
            sym = OK

        print(f'  {pair:<10}  {avg_pips:>10.2f}  {(mn or 0):>10.6f}  '
              f'{(mx or 0):>10.5f}  {n_null:>8}  {sym}')

    print(f'\n  ATR issues : {atr_issues}')
    print(f'  (Min/max over 21 years includes GFC, SNB, COVID extremes — only avg & NULL checked)')

    # =========================================================================
    # CHECK 7 — Fee / spread bleed-out risk
    # =========================================================================
    _hdr('CHECK 7 -- FEE / SPREAD BLEED-OUT RISK  (threshold: spread > 15% of ATR)')

    print(f'\n  {"PAIR":<10}  {"SPREAD(pip)":>11}  {"ATR(pip)":>9}  {"COST%":>6}  STATUS')
    print(f'  {"-"*10}  {"-"*11}  {"-"*9}  {"-"*6}  {"-"*14}')

    high_cost = []
    for pair in PAIRS:
        spread_pips = SPREADS.get(pair, 0.0)
        pip_scale   = PAIRS[pair]['pip_scale']
        avg_atr     = atr_map.get(pair)

        if not avg_atr:
            print(f'  {pair:<10}  {spread_pips:>11.1f}  {"N/A":>9}  {"N/A":>6}  {WARN}')
            continue

        atr_pips   = avg_atr * pip_scale
        cost_pct   = spread_pips / atr_pips * 100

        if cost_pct > 15.0:
            sym = FAIL
            high_cost.append((pair, cost_pct))
            global_issues += 1
        elif cost_pct > 10.0:
            sym = WARN
        else:
            sym = OK

        print(f'  {pair:<10}  {spread_pips:>11.1f}  {atr_pips:>9.2f}  {cost_pct:>5.1f}%  {sym}')

    if high_cost:
        print(f'\n  HIGH COST WARNING -- verify these pairs are worth trading:')
        for p, pct in high_cost:
            print(f'    {p}  ({pct:.1f}% of ATR per round-trip)')
    else:
        print(f'\n  No pairs exceed the 15% spread/ATR threshold.')

    # =========================================================================
    # CHECK 8 — Summary
    # =========================================================================
    _hdr('SUMMARY')

    pairs_ok = sum(
        1 for s in pair_stats.values()
        if s['total'] >= MIN_BARS_REQUIRED and s['n_train'] >= 2000 and s['n_null'] == 0
    )
    pairs_fail = len(pair_stats) - pairs_ok

    print(f'\n  Pairs in database  : {len(pair_stats)} / {len(PAIRS)}')
    print(f'  Pairs passing      : {pairs_ok}')
    print(f'  Pairs failing      : {pairs_fail}')
    print(f'  Total bars (all)   : {total_bars_all:,}')
    print(f'  Total issues found : {global_issues}')
    print()

    if test_contaminated:
        print(f'  ✗ TEST SET INTEGRITY: CONTAMINATED')
    else:
        print(f'  ✓ TEST SET INTEGRITY: CLEAN')

    print()
    if global_issues == 0:
        print(f'  ✓✓  DATA PIPELINE COMPLETE -- READY TO PROCEED')
    else:
        verdict = 'REVIEW BEFORE PROCEEDING' if not test_contaminated else 'DO NOT PROCEED'
        print(f'  ✗  ISSUES FOUND ({global_issues} total) -- {verdict}')
    print()

    conn.close()


if __name__ == '__main__':
    run_validation()
