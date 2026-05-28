"""
Gymnasium trading environment for the forex RL bot.
Implements Nick Shawn's S&R zone-based trading strategy.
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import gymnasium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zone_engine.sr_engine import SREngine
from data_pipeline.config import PAIRS, SPREADS

# Action indices
PASS         = 0
ENTER_LONG   = 1
ENTER_SHORT  = 2
HOLD         = 3
CLOSE        = 4
PARTIAL_CLOSE = 5


class ForexTradingEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for forex trading using S&R zones.
    Observation: 66-element float32 vector.
    Action: Discrete(6).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, db_path, pairs, split='train', window_size=252,
                 initial_equity=100_000.0,
                 date_start=None,
                 date_end=None,
                 use_reward_shaping: bool = False):
        super().__init__()
        self.db_path = db_path
        self.pairs = list(pairs)
        self.split = split
        self.window_size = window_size
        self.initial_equity = float(initial_equity)
        self.date_start = date_start
        self.date_end   = date_end

        self.action_space = gymnasium.spaces.Discrete(6)
        self.observation_space = gymnasium.spaces.Box(
            low=-10.0, high=10.0, shape=(66,), dtype=np.float32
        )

        # Loaded at init; never re-queried during episodes
        self._data: dict[str, pd.DataFrame] = {}
        self._zone_features: dict[str, list] = {}   # pair -> list[np.ndarray shape(48,)]
        self._usd_idx: dict[str, float] = {}
        self._jpy_idx: dict[str, float] = {}
        self._gbp_idx: dict[str, float] = {}
        self._risk_sentiment: dict[str, float] = {}

        self._load_data()
        self._precompute_zones()
        self._precompute_correlation()

        # Episode state — reset in reset()
        self._current_pair: str = self.pairs[0]
        self._episode_start: int = 0
        self._current_step: int = 0

        self._equity: float = self.initial_equity
        self._peak_equity: float = self.initial_equity
        self._in_trade: bool = False
        self._trade_direction: int = 0      # +1 long, -1 short, 0 none
        self._entry_price: float = 0.0
        self._stop_loss: float = 0.0
        self._take_profit: float = 0.0
        self._risk_amount: float = 0.0      # dollars risked at entry
        self._bars_in_trade: int = 0
        self._partial_closed: bool = False
        self._remaining_fraction: float = 1.0  # fraction of position still open

        self._rng = np.random.default_rng()

        # ── Reward shaping (off by default; enabled in final_trainer.py) ─────────
        self._use_reward_shaping  = use_reward_shaping
        self._ENTRY_ZONE_COEF     = 0.12   # max bonus for entry exactly at zone midpoint
        self._ZONE_DIST_CAP       = 1.5    # ATR units beyond which entry bonus is zero
        self._HOLD_WINNER_BONUS   = 0.008  # per-bar bonus while holding unrealized_r > 0.25
        self._EARLY_CUT_BONUS     = 0.025  # bonus for model-initiated close of losing trade

    # ── Data loading ────────────────────────────────────────────────────────

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        for pair in self.pairs:
            _where  = "pair = ? AND split_set = ? AND is_anomaly = 0"
            _params = [pair, self.split]
            if self.date_start:
                _where  += " AND date >= ?"
                _params.append(self.date_start)
            if self.date_end:
                _where  += " AND date <= ?"
                _params.append(self.date_end)

            df = pd.read_sql_query(
                f"""SELECT date, open, high, low, close, tick_volume,
                          tick_vol_zscore, atr_14, atr_200, spread_est
                   FROM raw_ohlcv
                   WHERE {_where}
                   ORDER BY date ASC""",
                conn, params=_params
            )
            if df.empty:
                self._data[pair] = df
                continue
            df['date'] = pd.to_datetime(df['date'])
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
            df['tick_vol_zscore'] = df['tick_vol_zscore'].fillna(0.0)
            df['atr_14']  = df['atr_14'].ffill().fillna(0.0)
            df['atr_200'] = df['atr_200'].ffill().fillna(0.0)
            df['spread_est'] = df['spread_est'].fillna(0.0)
            df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon … 4=Fri
            df = df.reset_index(drop=True)
            self._data[pair] = df
        conn.close()

    # ── Zone pre-computation ─────────────────────────────────────────────────

    def _precompute_zones(self):
        for pair in self.pairs:
            df = self._data[pair]
            if len(df) < 10:
                self._zone_features[pair] = [
                    np.zeros(48, dtype=np.float32) for _ in range(len(df))
                ]
                continue

            pip_scale = float(PAIRS.get(pair, {}).get('pip_scale', 10_000))
            engine_df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
                columns={'tick_volume': 'volume'}
            ).copy()

            engine = SREngine(engine_df, pip_scale=pip_scale)

            feats = []
            for i in range(len(df)):
                engine.step(i)
                zd = engine.get_nearest_zones(i, n_sup=2, n_res=2)
                sup = zd['sup_features'].flatten()   # (2,12) -> (24,)
                res = zd['res_features'].flatten()   # (2,12) -> (24,)
                feats.append(np.concatenate([sup, res]).astype(np.float32))

            self._zone_features[pair] = feats

    # ── Correlation pre-computation ─────────────────────────────────────────

    def _precompute_correlation(self):
        # Build rolling 20-bar return z-score per pair keyed by date string
        pair_ret_z: dict[str, dict[str, float]] = {}
        for pair in self.pairs:
            df = self._data[pair]
            if df.empty:
                continue
            log_ret = np.log(df['close'] / df['close'].shift(1))
            rm = log_ret.rolling(20).mean()
            rs = log_ret.rolling(20).std().replace(0.0, 1e-8)
            z = (log_ret - rm) / rs
            pair_ret_z[pair] = dict(zip(df['date_str'], z.fillna(0.0)))

        # Helper: weighted average z across a list of (pair, sign) tuples per date
        def _make_index(pair_sign_list: list) -> dict:
            all_dates: set = set()
            for p, _ in pair_sign_list:
                all_dates.update(pair_ret_z.get(p, {}).keys())
            idx = {}
            for d in all_dates:
                vals = [pair_ret_z[p].get(d, np.nan) * s
                        for p, s in pair_sign_list if p in pair_ret_z]
                valid = [v for v in vals if not np.isnan(v)]
                idx[d] = float(np.mean(valid)) if valid else 0.0
            return idx

        # USD strength (+1 = USD strong)
        # USD-quote pairs: EUR/USD, GBP/USD, AUD/USD, NZD/USD → USD strong = rate falls → sign -1
        # USD-base pairs: USD/JPY, USD/CHF, USD/CAD → USD strong = rate rises → sign +1
        usd_ps = [('EURUSD',-1),('GBPUSD',-1),('AUDUSD',-1),('NZDUSD',-1),
                  ('USDJPY',+1),('USDCHF',+1),('USDCAD',+1)]
        jpy_ps = [('USDJPY',-1),('EURJPY',-1),('GBPJPY',-1),('AUDJPY',-1),
                  ('CADJPY',-1),('CHFJPY',-1),('NZDJPY',-1)]
        gbp_ps = [('GBPUSD',+1),('EURGBP',-1),('GBPJPY',+1),
                  ('GBPAUD',+1),('GBPCAD',+1),('GBPNZD',+1)]

        available_usd = [(p, s) for p, s in usd_ps if p in pair_ret_z]
        available_jpy = [(p, s) for p, s in jpy_ps if p in pair_ret_z]
        available_gbp = [(p, s) for p, s in gbp_ps if p in pair_ret_z]

        self._usd_idx = _make_index(available_usd)
        self._jpy_idx = _make_index(available_jpy)
        self._gbp_idx = _make_index(available_gbp)

        # Risk sentiment: 10-bar AUD/JPY direction (+1 risk-on, -1 risk-off)
        risk: dict[str, float] = {}
        if 'AUDJPY' in self._data and not self._data['AUDJPY'].empty:
            df_aj = self._data['AUDJPY']
            roll_ret = df_aj['close'].pct_change(10)
            for d, v in zip(df_aj['date_str'], roll_ret):
                risk[d] = 1.0 if (not np.isnan(v) and v > 0) else -1.0
        self._risk_sentiment = risk

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        valid = [p for p in self.pairs
                 if len(self._data.get(p, [])) > self.window_size]
        if not valid:
            valid = [p for p in self.pairs if len(self._data.get(p, [])) > 1]
        if not valid:
            valid = self.pairs

        self._current_pair = str(self._rng.choice(valid))
        df = self._data[self._current_pair]
        max_start = max(1, len(df) - self.window_size)
        self._episode_start = int(self._rng.integers(0, max_start))
        self._current_step = 0

        self._equity = self.initial_equity
        self._peak_equity = self.initial_equity
        self._reset_trade_state()

        obs = self._get_observation()
        info = {
            'pair': self._current_pair,
            'episode_start': self._episode_start,
            'equity': self._equity,
        }
        return obs, info

    def step(self, action: int):
        assert self._current_pair is not None, "Call reset() before step()"

        # ── Snapshot state before action (used by reward shaping at end) ─────────
        _pre_in_trade = self._in_trade
        _bar_before   = self._bar_idx()
        _close_before = float(self._current_row()['close'])
        _zf_before    = self._zone_features[self._current_pair][_bar_before].copy()

        row = self._current_row()
        close   = float(row['close'])
        atr14   = max(float(row['atr_14']),  1e-8)
        spread_price = float(row['spread_est']) / float(
            PAIRS.get(self._current_pair, {}).get('pip_scale', 10_000)
        )

        reward = 0.0
        info: dict = {}

        # ── Auto stop/tp check (only when already in trade) ─────────────────
        if self._in_trade:
            hit_sl, hit_tp = self._check_stop_tp(row)
            if hit_sl or hit_tp:
                exit_px = self._stop_loss if hit_sl else self._take_profit
                # spread cost on exit
                exit_px -= spread_price * self._trade_direction
                realized_r = self._pnl_in_r(exit_px)
                self._equity += realized_r * self._risk_amount
                self._peak_equity = max(self._peak_equity, self._equity)
                reward = float(realized_r)
                info['exit_reason'] = 'stop_loss' if hit_sl else 'take_profit'
                info['realized_r']  = realized_r
                self._reset_trade_state()
                self._advance()
                obs = self._get_observation()
                terminated = self._equity < self.initial_equity * 0.5
                truncated  = self._current_step >= self.window_size
                info['equity'] = self._equity
                return obs, reward, terminated, truncated, info

        # ── Process agent action ─────────────────────────────────────────────
        if action in (PASS, HOLD):
            if self._in_trade:
                ur = self._unrealized_r(close)
                reward = 0.01 if ur > 0 else (-0.02 if ur < -0.5 else 0.0)

        elif action in (ENTER_LONG, ENTER_SHORT):
            if self._in_trade:
                reward = -0.05  # invalid: already in trade
            else:
                direction = 1 if action == ENTER_LONG else -1
                dist = self._nearest_zone_dist_atr(direction)
                reward = -0.1 if dist > 1.5 else 0.0
                self._enter_trade(direction, close, atr14, spread_price)

        elif action == CLOSE:
            if not self._in_trade:
                reward = -0.05
            else:
                exit_px = close - spread_price * self._trade_direction
                realized_r = self._pnl_in_r(exit_px)
                self._equity += realized_r * self._risk_amount
                self._peak_equity = max(self._peak_equity, self._equity)
                reward = float(realized_r)
                info['exit_reason'] = 'manual_close'
                info['realized_r']  = realized_r
                self._reset_trade_state()

        elif action == PARTIAL_CLOSE:
            if not self._in_trade or self._partial_closed:
                reward = -0.05
            else:
                ur = self._unrealized_r(close)
                reward = float(0.5 * ur)
                exit_px = close - spread_price * self._trade_direction
                partial_r = self._pnl_in_r(exit_px) * 0.5
                self._equity += partial_r * self._risk_amount
                self._peak_equity = max(self._peak_equity, self._equity)
                # Move stop to breakeven, halve remaining position
                self._stop_loss = self._entry_price
                self._partial_closed = True
                self._remaining_fraction = 0.5

        if self._in_trade:
            self._bars_in_trade += 1

        self._advance()
        obs = self._get_observation()
        terminated = self._equity < self.initial_equity * 0.5
        truncated  = self._current_step >= self.window_size
        info.update({'equity': self._equity, 'pair': self._current_pair,
                     'step': self._current_step})

        # ── Reward shaping ────────────────────────────────────────────────────────
        if self._use_reward_shaping:
            # 1. Zone entry bonus — reward entering near S&R zone midpoint
            if (not _pre_in_trade) and self._in_trade:
                dist = (abs(float(_zf_before[1]))
                        if self._trade_direction == 1
                        else abs(float(_zf_before[25])))
                if dist < self._ZONE_DIST_CAP:
                    reward += self._ENTRY_ZONE_COEF * max(0.0, 1.0 - dist / self._ZONE_DIST_CAP)

            # 2. Hold winner — tiny per-bar bonus for holding a profitable trade
            if _pre_in_trade and self._in_trade:
                if self._unrealized_r(_close_before) > 0.25:
                    reward += self._HOLD_WINNER_BONUS

            # 3. Early cut — bonus for model-initiated close of a losing trade
            if _pre_in_trade and (not self._in_trade) and action == CLOSE:
                if info.get('realized_r', 0.0) < 0.0:
                    reward += self._EARLY_CUT_BONUS

            # 4. Missed-zone penalty — penalise PASS/HOLD when near a valid zone
            if (not _pre_in_trade) and action in (PASS, HOLD):
                sup_valid = float(np.sum(np.abs(_zf_before[0:12])))  > 1e-6
                res_valid = float(np.sum(np.abs(_zf_before[24:36]))) > 1e-6
                sup_dist  = abs(float(_zf_before[1]))  if sup_valid  else 999.0
                res_dist  = abs(float(_zf_before[25])) if res_valid  else 999.0
                min_dist  = min(sup_dist, res_dist)
                if min_dist < 1.0:
                    reward -= 0.05 * max(0.0, 1.0 - min_dist)

        return obs, reward, terminated, truncated, info

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        bi   = self._bar_idx()
        pair = self._current_pair
        row  = self._data[pair].iloc[bi]

        close  = float(row['close'])
        atr14  = max(float(row['atr_14']),  1e-8)
        atr200 = max(float(row['atr_200']), 1e-8)
        tick_z = float(row['tick_vol_zscore'])
        dow    = float(row['day_of_week']) / 4.0   # [0, 1]

        # Zone features [48]
        zf = self._zone_features[pair][bi].copy()

        # Distances to zone midpoints (already ATR-normalised in zf)
        # zf[1]  = dist sup1 mid, zf[13] = dist sup2 mid
        # zf[25] = dist res1 mid (negative = above price), zf[37] = dist res2 mid
        sup1_d = float(zf[1])
        sup2_d = float(zf[13])
        res1_d = float(zf[25])
        res2_d = float(zf[37])

        # Price action context [8]
        price_ctx = np.array([
            sup1_d,
            sup2_d,
            res1_d,
            res2_d,
            atr14  / close,
            atr200 / close,
            tick_z,
            dow,
        ], dtype=np.float32)

        # Unrealized R
        ur = float(np.clip(self._unrealized_r(close), -3.0, 3.0))

        # Trade state [6]
        trade_state = np.array([
            1.0 if self._in_trade else 0.0,
            (self._trade_direction + 1) / 2.0,    # -1→0, 0→0.5, +1→1
            ur,
            min(self._bars_in_trade / 20.0, 5.0),
            self._equity / self.initial_equity,
            1.0 - self._equity / max(self._peak_equity, 1e-8),
        ], dtype=np.float32)

        # Correlation flags [4]
        date_str = str(row['date_str'])
        corr = np.array([
            float(np.tanh(self._usd_idx.get(date_str, 0.0))),
            float(np.tanh(self._jpy_idx.get(date_str, 0.0))),
            float(np.tanh(self._gbp_idx.get(date_str, 0.0))),
            float(np.clip(self._risk_sentiment.get(date_str, 0.0), -1.0, 1.0)),
        ], dtype=np.float32)

        obs = np.concatenate([zf, price_ctx, trade_state, corr])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _bar_idx(self) -> int:
        df = self._data[self._current_pair]
        return min(self._episode_start + self._current_step, len(df) - 1)

    def _current_row(self):
        return self._data[self._current_pair].iloc[self._bar_idx()]

    def _advance(self):
        self._current_step += 1

    def _check_stop_tp(self, row) -> tuple:
        h, l_ = float(row['high']), float(row['low'])
        if self._trade_direction == 1:    # long
            return l_ <= self._stop_loss, h >= self._take_profit
        else:                             # short
            return h >= self._stop_loss,  l_ <= self._take_profit

    def _nearest_zone_dist_atr(self, direction: int) -> float:
        zf = self._zone_features[self._current_pair][self._bar_idx()]
        if direction == 1:      # long → nearest support zone
            feat = zf[0:12]
            if np.sum(np.abs(feat)) < 1e-6:
                return 999.0
            return abs(float(feat[1]))
        else:                   # short → nearest resistance zone
            feat = zf[24:36]
            if np.sum(np.abs(feat)) < 1e-6:
                return 999.0
            return abs(float(feat[1]))

    def _zone_stop_tp(self, direction: int, close: float, atr14: float):
        """Derive entry/stop/tp from nearest zone feature vector."""
        zf = self._zone_features[self._current_pair][self._bar_idx()]
        if direction == 1:
            feat = zf[0:12]
        else:
            feat = zf[24:36]

        dist_mid  = float(feat[1])
        dist_near = float(feat[2])

        # Reconstruct zone geometry
        midpoint  = close - dist_mid  * atr14
        near_edge = close - dist_near * atr14
        far_edge  = 2.0 * midpoint - near_edge

        if direction == 1:  # long: near=top, far=btm
            sl = far_edge - 0.1 * atr14
            tp = close + abs(close - sl)
        else:               # short: near=btm, far=top
            sl = far_edge + 0.1 * atr14
            tp = close - abs(sl - close)

        # Fallback: if stop is on the wrong side or too tight, use ATR-based stop
        if direction == 1 and sl >= close:
            sl = close - 1.5 * atr14
            tp = close + abs(close - sl)
        if direction == -1 and sl <= close:
            sl = close + 1.5 * atr14
            tp = close - abs(sl - close)

        return close, sl, tp

    def _enter_trade(self, direction: int, close: float, atr14: float,
                     spread_price: float):
        entry, sl, tp = self._zone_stop_tp(direction, close, atr14)
        # Apply spread on entry (widens effective cost)
        entry_eff = entry + direction * spread_price

        stop_dist = abs(entry_eff - sl)
        if stop_dist < 1e-8:
            stop_dist = atr14

        risk_amount = self._equity * 0.10
        # Deduct entry spread cost from equity immediately
        position_units = risk_amount / stop_dist
        spread_cost = spread_price * position_units
        self._equity -= spread_cost
        self._peak_equity = max(self._peak_equity, self._equity)

        self._in_trade         = True
        self._trade_direction  = direction
        self._entry_price      = entry_eff
        self._stop_loss        = sl
        self._take_profit      = tp
        self._risk_amount      = risk_amount
        self._bars_in_trade    = 0
        self._partial_closed   = False
        self._remaining_fraction = 1.0

    def _pnl_in_r(self, exit_price: float) -> float:
        """P&L in R-multiples for the remaining fraction of the position."""
        risk_per_unit = abs(self._entry_price - self._stop_loss)
        if risk_per_unit < 1e-8:
            return 0.0
        raw = (exit_price - self._entry_price) * self._trade_direction
        return raw / risk_per_unit * self._remaining_fraction

    def _unrealized_r(self, close: float) -> float:
        if not self._in_trade:
            return 0.0
        return self._pnl_in_r(close)

    def _reset_trade_state(self):
        self._in_trade           = False
        self._trade_direction    = 0
        self._entry_price        = 0.0
        self._stop_loss          = 0.0
        self._take_profit        = 0.0
        self._risk_amount        = 0.0
        self._bars_in_trade      = 0
        self._partial_closed     = False
        self._remaining_fraction = 1.0

    # ── Properties for external monitoring ───────────────────────────────────

    @property
    def current_pair(self) -> str:
        return self._current_pair

    @property
    def current_step(self) -> int:
        return self._current_step

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, mode='human'):
        if self._current_pair is None:
            print("Not initialized — call reset() first.")
            return
        row = self._current_row()
        print(
            f"[{self._current_pair}] step={self._current_step}/{self.window_size} | "
            f"close={float(row['close']):.5f} | equity=${self._equity:,.2f} | "
            f"in_trade={self._in_trade} dir={self._trade_direction}"
        )
