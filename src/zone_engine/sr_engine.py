"""
LuxAlgo S&R Pro Toolkit — Python Port + Nick Shawn Extensions
Translated from Pine Script v6, with additions for the RL trading bot.

Zone fields match the Pine Script type `level` exactly, plus:
  - rapid_test_count   : entries in the last RECENCY_WINDOW bars (exhaustion)
  - bounce_quality     : avg max-favorable-excursion (pips) per touch
  - bounce_history     : list of (bar_of_touch, mfe_pips) tuples
  - is_break_retest    : True if zone changed from resistance→support or vice versa
  - grade              : 'A+' / 'A' / 'B' computed from quality factors
  - last_entry_bar     : bar index of the most recent entry (for recency scoring)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

# ── Tunable Parameters (these are what the RL optimiser sweeps) ────────────────
DEFAULT_PARAMS = {
    # Detection
    "method": "Donchian",          # 'Pivots' | 'Donchian' | 'CSID' | 'ZigZag'
    "sensitivity": 10,             # lookback for Pivots/Donchian/CSID; % deviation for ZigZag
    # Zone sizing
    "atr_period": 200,
    "atr_mult": 0.5,               # zone depth = basePrice ± ATR * atr_mult
    "security_mult": 0.0,          # breakout buffer on far side
    # Overlap
    "overlap_mode": "hide_old",    # 'none' | 'merge' | 'hide_old' | 'hide_young'
    "max_active": 5,               # max simultaneously visible unmitigated zones
    # Minimum filters
    "min_entries": 0,
    "min_strength": 0,             # additional swing points inside zone
    "min_sweeps": 0,
    "min_volume": 0.0,
    "min_duration": 0,             # bars
    # Nick-specific
    "recency_window": 25,          # bars for rapid succession count (exhaustion)
    "exhaustion_threshold": 4,     # rapid tests that flag a zone as exhausted
    "min_bounce_pips": 15,         # MFE below this = poor bounce quality
    "grade_a_plus_mfe": 40,        # avg MFE (pips) to qualify as A+
    "grade_a_mfe": 20,             # avg MFE to qualify as A  (else B)
}

PARAM_SPACE = {
    "sensitivity":          [3, 5, 7, 10, 14, 20, 30],
    "atr_mult":             [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "security_mult":        [0.0, 0.1, 0.2, 0.3, 0.5],
    "max_active":           [3, 4, 5, 6, 8],
    "min_entries":          [0, 1, 2, 3],
    "min_sweeps":           [0, 1, 2],
    "recency_window":       [15, 20, 25, 30],
    "exhaustion_threshold": [3, 4, 5],
    "grade_a_plus_mfe":     [30, 40, 50, 60],
    "grade_a_mfe":          [15, 20, 25],
    "method":               ["Pivots", "Donchian", "ZigZag"],
    "overlap_mode":         ["merge", "hide_old", "hide_young"],
}


# ── Zone Data Object ───────────────────────────────────────────────────────────
@dataclass
class Zone:
    # Core fields (matching Pine Script `level` type)
    top: float
    btm: float
    base_price: float
    start_bar: int
    is_support: bool
    is_mitigated: bool = False
    is_user_hidden: bool = False
    mitigation_bar: Optional[int] = None
    entries: int = 0
    strength: int = 0        # swing points that formed inside zone while active
    sweeps: int = 0
    traded_volume: float = 0.0

    # Nick Shawn extensions
    bounce_history: List[Tuple[int, float]] = field(default_factory=list)
    # each element = (bar_of_touch, max_favorable_excursion_in_pips)
    rapid_test_count: int = 0      # entries in the last recency_window bars
    last_entry_bar: int = -999
    is_break_retest: bool = False   # resistance→support or support→resistance flip

    @property
    def midpoint(self) -> float:
        return (self.top + self.btm) / 2

    @property
    def width_pips(self) -> float:
        return (self.top - self.btm) * 10_000  # assumes 4-decimal pair; JPY: ×100

    @property
    def duration(self) -> int:
        if self.is_mitigated and self.mitigation_bar is not None:
            return self.mitigation_bar - self.start_bar
        return 0  # caller adds bar_index - start_bar for live zones

    @property
    def bounce_quality(self) -> float:
        """Average max-favorable-excursion (pips) across all touch events."""
        if not self.bounce_history:
            return 0.0
        return float(np.mean([mfe for _, mfe in self.bounce_history]))

    def grade(self, params: dict) -> str:
        """
        A+ : bounce_quality >= grade_a_plus_mfe  AND entries >= 2
        A  : bounce_quality >= grade_a_mfe
        B  : everything else (still tradeable but lower risk)
        """
        bq = self.bounce_quality
        if bq >= params["grade_a_plus_mfe"] and self.entries >= 2:
            return "A+"
        elif bq >= params["grade_a_mfe"]:
            return "A"
        return "B"

    def exhaustion_score(self, current_bar: int, params: dict) -> float:
        """
        Returns a 0-1 score of exhaustion pressure.
        0 = fresh zone, 1 = fully exhausted.
        Based on rapid_test_count vs threshold.
        """
        return min(1.0, self.rapid_test_count / max(1, params["exhaustion_threshold"]))

    def is_exhausted(self, current_bar: int, params: dict) -> bool:
        return self.rapid_test_count >= params["exhaustion_threshold"]

    def recency_score(self, current_bar: int, decay_bars: int = 100) -> float:
        """1.0 = just formed, decays toward 0 as zone ages."""
        age = current_bar - self.start_bar
        return max(0.0, 1.0 - age / decay_bars)

    def to_feature_vector(self, current_bar: int, current_price: float,
                          atr: float, params: dict) -> np.ndarray:
        """
        Convert zone to a fixed-length feature vector for the RL state.
        Returns 12-element float array.
        """
        atr_safe = max(atr, 1e-8)
        dur = current_bar - self.start_bar
        dist_to_mid = (current_price - self.midpoint) / atr_safe
        dist_to_near = (current_price - (self.btm if self.is_support else self.top)) / atr_safe
        g = self.grade(params)
        grade_enc = {"A+": 1.0, "A": 0.6, "B": 0.2}.get(g, 0.0)
        return np.array([
            float(self.is_support),           # 0: 1=support, 0=resistance
            dist_to_mid,                      # 1: distance to midpoint (ATR-normalized)
            dist_to_near,                     # 2: distance to near edge
            self.entries / 10.0,              # 3: entries (normalized)
            self.sweeps / 5.0,                # 4: sweeps (normalized)
            self.bounce_quality / 100.0,      # 5: avg MFE quality (normalized)
            self.exhaustion_score(current_bar, params),  # 6: exhaustion 0-1
            self.recency_score(current_bar),  # 7: zone freshness
            grade_enc,                        # 8: A+=1.0, A=0.6, B=0.2
            min(dur / 200.0, 1.0),            # 9: age (normalized, capped)
            float(self.is_break_retest),      # 10: break-and-retest flag
            self.width_pips / 100.0,          # 11: zone width (normalized)
        ], dtype=np.float32)


# ── ATR Calculator ─────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 200) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    # Fall back to cumulative average when ATR is NaN
    cum_range = (high - low).cumsum()
    bar_count  = pd.Series(range(1, len(df)+1), index=df.index)
    fallback    = cum_range / bar_count
    return atr.fillna(fallback)


# ── Swing Detection ─────────────────────────────────────────────────────────────
def detect_swings_pivots(df: pd.DataFrame, sensitivity: int
                         ) -> Tuple[pd.Series, pd.Series]:
    """Standard pivot high / pivot low with left=right=sensitivity."""
    n = len(df)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    s = sensitivity
    for i in range(s, n - s):
        h = df["high"].iloc[i]
        l_ = df["low"].iloc[i]
        if h == df["high"].iloc[i-s:i+s+1].max():
            ph.iloc[i] = h
        if l_ == df["low"].iloc[i-s:i+s+1].min():
            pl.iloc[i] = l_
    return ph, pl


def detect_swings_donchian(df: pd.DataFrame, sensitivity: int
                           ) -> Tuple[pd.Series, pd.Series]:
    """
    Alternating state machine: direction flips when price makes a new
    Donchian high (bullish) or new Donchian low (bearish).
    Confirmed extreme is emitted when direction changes.
    """
    n = len(df)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    donch_os = 0
    donch_val = np.nan
    donch_val_loc = -1

    roll_high = df["high"].rolling(sensitivity).max()
    roll_low  = df["low"].rolling(sensitivity).min()

    for i in range(1, n):
        rh  = roll_high.iloc[i]
        rl  = roll_low.iloc[i]
        rh1 = roll_high.iloc[i-1]
        rl1 = roll_low.iloc[i-1]
        new_os = donch_os

        if rh > rh1:
            new_os = 1
        elif rl < rl1:
            new_os = -1

        if new_os != donch_os:
            if new_os == 1:       # was bearish, now bullish → emit low
                if not np.isnan(donch_val):
                    pl.iloc[donch_val_loc] = donch_val
                donch_val = df["high"].iloc[i]
                donch_val_loc = i
            else:                 # was bullish, now bearish → emit high
                if not np.isnan(donch_val):
                    ph.iloc[donch_val_loc] = donch_val
                donch_val = df["low"].iloc[i]
                donch_val_loc = i
            donch_os = new_os
        else:
            h = df["high"].iloc[i]
            l_ = df["low"].iloc[i]
            if donch_os == 1 and h >= (donch_val if not np.isnan(donch_val) else -1e10):
                donch_val = h
                donch_val_loc = i
            elif donch_os == -1 and l_ <= (donch_val if not np.isnan(donch_val) else 1e10):
                donch_val = l_
                donch_val_loc = i
    return ph, pl


def detect_swings_csid(df: pd.DataFrame, sensitivity: int
                       ) -> Tuple[pd.Series, pd.Series]:
    """
    Consecutive-candle state: after N consecutive bull candles emit the highest
    high; after N consecutive bear candles emit the lowest low.
    """
    n = len(df)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    bull = 0; bear = 0
    roll_h = df["high"].rolling(sensitivity).max()
    roll_l = df["low"].rolling(sensitivity).min()
    hbars   = df["high"].rolling(sensitivity).apply(lambda x: -np.argmax(x[::-1]), raw=True).astype(int)
    lbars   = df["low"].rolling(sensitivity).apply(lambda x: -np.argmin(x[::-1]),  raw=True).astype(int)

    for i in range(n):
        bull = bull + 1 if df["close"].iloc[i] > df["open"].iloc[i] else 0
        bear = bear + 1 if df["close"].iloc[i] < df["open"].iloc[i] else 0
        if bull == sensitivity:
            loc = i + int(hbars.iloc[i])
            if 0 <= loc < n:
                ph.iloc[loc] = roll_h.iloc[i]
        if bear == sensitivity:
            loc = i + int(lbars.iloc[i])
            if 0 <= loc < n:
                pl.iloc[loc] = roll_l.iloc[i]
    return ph, pl


def detect_swings_zigzag(df: pd.DataFrame, sensitivity: float
                         ) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag by percentage deviation.  Emit confirmed high/low when price
    reverses by (close * sensitivity / 100) from the running extreme.
    """
    n = len(df)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    zz_high = zz_low = np.nan
    zz_high_bar = zz_low_bar = 0
    zz_dir = 0

    for i in range(n):
        close = df["close"].iloc[i]
        high  = df["high"].iloc[i]
        low   = df["low"].iloc[i]
        dev   = close * sensitivity / 100.0

        if zz_dir == 0:
            zz_high = high; zz_low = low
            zz_high_bar = zz_low_bar = i
            zz_dir = 1 if close > df["open"].iloc[i] else -1
        elif zz_dir == 1:
            if high > zz_high:
                zz_high = high; zz_high_bar = i
            elif low < zz_high - dev:
                ph.iloc[zz_high_bar] = zz_high
                zz_low = low; zz_low_bar = i; zz_dir = -1
        else:
            if low < zz_low:
                zz_low = low; zz_low_bar = i
            elif high > zz_low + dev:
                pl.iloc[zz_low_bar] = zz_low
                zz_high = high; zz_high_bar = i; zz_dir = 1
    return ph, pl


# ── Main Zone Engine ────────────────────────────────────────────────────────────
class SREngine:
    """
    Full Python port of LuxAlgo S&R Pro Toolkit + Nick Shawn extensions.
    Processes bars sequentially (call .step(i) for each bar).
    """

    def __init__(self, df: pd.DataFrame, params: dict = None,
                 pip_scale: float = 10_000.0):
        """
        df          : DataFrame with columns [open, high, low, close, volume]
        params      : overrides for DEFAULT_PARAMS
        pip_scale   : 10_000 for 4-decimal pairs, 100 for JPY pairs
        """
        self.df  = df.reset_index(drop=True)
        self.p   = {**DEFAULT_PARAMS, **(params or {})}
        self.pip = pip_scale
        self.n   = len(df)

        # Pre-compute ATR and swing series
        self.atr = compute_atr(df, self.p["atr_period"])
        self.ph, self.pl = self._compute_swings()

        # Zone state
        self.zones: List[Zone] = []

        # Stats (mirrors Pine Script dashboard vars)
        self.total_sup = 0; self.total_res = 0
        self.sup_mitigated = 0; self.res_mitigated = 0
        self.sup_dur = 0; self.res_dur = 0
        self.sup_sweeps = 0; self.res_sweeps = 0
        self.sup_cum_vol = 0.0; self.res_cum_vol = 0.0

        # Tracking previous swing for break-retest detection
        self._recently_broken_sup: List[float] = []  # price levels
        self._recently_broken_res: List[float] = []

    def _compute_swings(self):
        m = self.p["method"]
        s = self.p["sensitivity"]
        if m == "Pivots":
            return detect_swings_pivots(self.df, int(s))
        elif m == "Donchian":
            return detect_swings_donchian(self.df, int(s))
        elif m == "CSID":
            return detect_swings_csid(self.df, int(s))
        elif m == "ZigZag":
            return detect_swings_zigzag(self.df, float(s))
        else:
            raise ValueError(f"Unknown method: {m}")

    def _pips(self, price_diff: float) -> float:
        return abs(price_diff) * self.pip

    def _create_zone(self, base: float, bar_i: int, is_support: bool) -> bool:
        """Create a zone from a detected swing point. Returns True if zone was added."""
        atr = self.atr.iloc[bar_i]
        atr_d = self.p["atr_mult"] * atr
        atr_b = self.p["security_mult"] * atr

        if is_support:
            top = base + atr_d
            btm = base - atr_b
        else:
            top = base + atr_b
            btm = base - atr_d

        # Detect break-and-retest: was this price level recently a broken zone
        # of the *opposite* type?
        is_brt = False
        if is_support:
            is_brt = any(abs(lv - base) < atr_d for lv in self._recently_broken_res)
        else:
            is_brt = any(abs(lv - base) < atr_d for lv in self._recently_broken_sup)

        # Overlap handling
        should_add = True
        om = self.p["overlap_mode"]
        active = [z for z in self.zones if not z.is_mitigated
                  and not z.is_user_hidden and z.is_support == is_support]
        for z in active:
            overlaps = max(btm, z.btm) < min(top, z.top)
            if overlaps:
                if om == "hide_old":
                    should_add = False
                    break
                elif om == "hide_young":
                    z.is_user_hidden = True
                elif om == "merge":
                    z.top = max(z.top, top)
                    z.btm = min(z.btm, btm)
                    should_add = False
                    break
                # "none" = add regardless

        if should_add:
            z_new = Zone(top=top, btm=btm, base_price=base,
                         start_bar=bar_i, is_support=is_support,
                         is_break_retest=is_brt)
            self.zones.insert(0, z_new)
            if is_support: self.total_sup += 1
            else:          self.total_res += 1
            return True
        return False

    def step(self, i: int) -> dict:
        """
        Process bar i. Returns a dict of events/observations for this bar.
        Call this in your RL environment's step() for each bar.
        """
        row = self.df.iloc[i]
        o, h, l_, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        atr = self.atr.iloc[i]

        events = {
            "bar": i,
            "close": c,
            "atr": atr,
            "zone_entered": [],   # list of Zone objects price just entered
            "zone_exited": [],    # list of Zone objects price just left
            "zone_mitigated": [], # list of Zone objects just broken/mitigated
            "new_zones": [],      # list of newly created Zone objects
        }

        # ── Create new zones from confirmed swings ──────────────────────────
        if not np.isnan(self.ph.iloc[i]):
            if self._create_zone(self.ph.iloc[i], i, is_support=False):
                events["new_zones"].append(self.zones[0])

        if not np.isnan(self.pl.iloc[i]):
            if self._create_zone(self.pl.iloc[i], i, is_support=True):
                events["new_zones"].append(self.zones[0])

        # ── Cap array size ──────────────────────────────────────────────────
        if len(self.zones) > 200:
            self.zones = self.zones[:200]

        # ── Process each zone ───────────────────────────────────────────────
        p = self.p
        rw = p["recency_window"]

        for z in self.zones:
            if z.is_mitigated:
                continue

            # Mitigation check (zone broken)
            if z.is_support and c < z.btm:
                z.is_mitigated = True
                z.mitigation_bar = i
                self.sup_mitigated += 1
                self.sup_dur += (i - z.start_bar)
                self.sup_cum_vol += z.traded_volume
                self._recently_broken_sup.append(z.base_price)
                if len(self._recently_broken_sup) > 20:
                    self._recently_broken_sup.pop(0)
                events["zone_mitigated"].append(z)
                continue
            elif not z.is_support and c > z.top:
                z.is_mitigated = True
                z.mitigation_bar = i
                self.res_mitigated += 1
                self.res_dur += (i - z.start_bar)
                self.res_cum_vol += z.traded_volume
                self._recently_broken_res.append(z.base_price)
                if len(self._recently_broken_res) > 20:
                    self._recently_broken_res.pop(0)
                events["zone_mitigated"].append(z)
                continue

            # Volume accumulation while price overlaps zone
            if h >= z.btm and l_ <= z.top:
                z.traded_volume += v

            # Strength: new swing points formed inside zone while active
            if not np.isnan(self.ph.iloc[i]) and z.btm <= self.ph.iloc[i] <= z.top:
                z.strength += 1
            if not np.isnan(self.pl.iloc[i]) and z.btm <= self.pl.iloc[i] <= z.top:
                z.strength += 1

            # ── Entry tracking ──────────────────────────────────────────────
            just_entered = False
            if z.is_support:
                if l_ <= z.top and c >= z.btm:
                    z.entries += 1
                    just_entered = True
                    events["zone_entered"].append(z)
                if l_ < z.btm and min(c, o) > z.btm:
                    z.sweeps += 1
                    self.sup_sweeps += 1
            else:
                if h >= z.btm and c <= z.top:
                    z.entries += 1
                    just_entered = True
                    events["zone_entered"].append(z)
                if h > z.top and max(c, o) < z.top:
                    z.sweeps += 1
                    self.res_sweeps += 1

            # ── Rapid succession count (exhaustion) ────────────────────────
            if just_entered:
                z.last_entry_bar = i
            # Count entries from bounce_history in the last rw bars
            z.rapid_test_count = sum(
                1 for (b, _) in z.bounce_history
                if i - rw <= b <= i
            ) + (1 if just_entered else 0)

            # ── Bounce quality (MFE) tracking ──────────────────────────────
            # For each entry event, look ahead is not possible in live mode.
            # Instead we update MFE of the PREVIOUS touch when price leaves zone.
            # We track it lazily: after price has moved back away from zone,
            # compute MFE of the last touch from the maximum reached since entry.
            if just_entered:
                z.bounce_history.append([i, 0.0])  # placeholder, MFE updated below

            # Update MFE for the most recent touch entry
            if z.bounce_history:
                last_touch_bar, last_mfe = z.bounce_history[-1]
                if z.is_support:
                    # MFE = max high since touch bar minus midpoint
                    mfe = self._pips(h - z.midpoint) if h > z.midpoint else 0.0
                else:
                    mfe = self._pips(z.midpoint - l_) if l_ < z.midpoint else 0.0
                z.bounce_history[-1][1] = max(last_mfe, mfe)

        # ── Filter to active zones for RL observation ───────────────────────
        active_zones = self._get_active_zones(i)
        events["active_zones"] = active_zones

        return events

    def _get_active_zones(self, current_bar: int) -> List[Zone]:
        """
        Return active (unmitigated, meets minimum filters) zones,
        sorted by proximity to current price.
        """
        p = self.p
        c = self.df["close"].iloc[current_bar]
        active = []
        count = 0

        for z in self.zones:
            if z.is_mitigated or z.is_user_hidden:
                continue
            dur = current_bar - z.start_bar
            meets = (
                z.entries >= p["min_entries"] and
                z.strength >= p["min_strength"] and
                z.sweeps >= p["min_sweeps"] and
                z.traded_volume >= p["min_volume"] and
                dur >= p["min_duration"]
            )
            if meets and count < p["max_active"]:
                active.append(z)
                count += 1

        # Sort: nearest zone to current price first
        active.sort(key=lambda z: abs(z.midpoint - c))
        return active

    def get_nearest_zones(self, current_bar: int,
                          n_sup: int = 2, n_res: int = 2) -> dict:
        """
        Convenience function: get the N nearest support zones below price
        and N nearest resistance zones above price.
        Returns feature vectors ready for the RL state.
        """
        c = self.df["close"].iloc[current_bar]
        atr = self.atr.iloc[current_bar]
        p = self.p

        active = self._get_active_zones(current_bar)
        supports = [z for z in active if z.is_support and z.midpoint < c]
        resists  = [z for z in active if not z.is_support and z.midpoint > c]
        # nearest first
        supports.sort(key=lambda z: c - z.midpoint)
        resists.sort(key=lambda z: z.midpoint - c)

        # Pad with zero vectors if not enough zones
        null_vec = np.zeros(12, dtype=np.float32)
        sup_feats = [z.to_feature_vector(current_bar, c, atr, p) for z in supports[:n_sup]]
        res_feats = [z.to_feature_vector(current_bar, c, atr, p) for z in resists[:n_res]]
        while len(sup_feats) < n_sup: sup_feats.append(null_vec.copy())
        while len(res_feats) < n_res: res_feats.append(null_vec.copy())

        return {
            "support_zones":    supports[:n_sup],
            "resistance_zones": resists[:n_res],
            "sup_features":     np.stack(sup_feats),   # shape (n_sup, 12)
            "res_features":     np.stack(res_feats),   # shape (n_res, 12)
        }

    def run_all(self) -> List[dict]:
        """Process all bars sequentially. Returns list of per-bar event dicts."""
        return [self.step(i) for i in range(self.n)]

    def stats(self) -> dict:
        """Summary stats matching LuxAlgo dashboard output."""
        total = self.total_sup + self.total_res
        mit   = self.sup_mitigated + self.res_mitigated
        return {
            "total_support":    self.total_sup,
            "total_resistance": self.total_res,
            "sup_mitigated":    self.sup_mitigated,
            "res_mitigated":    self.res_mitigated,
            "mitigation_rate":  mit / max(1, total),
            "avg_dur_sup":      self.sup_dur / max(1, self.sup_mitigated),
            "avg_dur_res":      self.res_dur / max(1, self.res_mitigated),
            "total_sweeps":     self.sup_sweeps + self.res_sweeps,
        }
