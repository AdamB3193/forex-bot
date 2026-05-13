"""
RiskGuardWrapper: hard safety rules over ForexTradingEnv that the PPO agent cannot override.
"""

import numpy as np
import gymnasium

PASS          = 0
ENTER_LONG    = 1
ENTER_SHORT   = 2
HOLD          = 3
CLOSE         = 4
PARTIAL_CLOSE = 5

_ENTRY_ACTIONS = (ENTER_LONG, ENTER_SHORT)

_USD_PAIRS = frozenset({'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD'})
_JPY_PAIRS = frozenset({'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY'})


class RiskGuardWrapper(gymnasium.Wrapper):
    """
    Wraps ForexTradingEnv and enforces five hard safety rules.

    Rule 1 — Daily Loss Limit:         block entries when daily realized R <= -3.0
    Rule 2 — Consecutive Loss Breaker: block entries for 5 bars after 3 consecutive losses
    Rule 3 — Correlation Guard:        block entries on USD/JPY pairs when those indices spike
    Rule 4 — Max Bars Hard Stop:       force CLOSE when trade open > 10 bars
    Rule 5 — Equity Curve Filter:      block entries when equity > 15% below 20-bar rolling avg
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialise (reset() will also call this path)
        self._blocked_reason_val: str = 'none'
        self._rules_triggered_val: dict = {
            'rule1': 0, 'rule2': 0, 'rule3': 0, 'rule4': 0, 'rule5': 0
        }
        self._daily_realized_r: float = 0.0
        self._current_day: str | None = None
        self._consecutive_losses: int = 0
        self._cooldown_bars_remaining: int = 0
        self._equity_history: list[float] = []

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._blocked_reason_val = 'none'
        self._rules_triggered_val = {
            'rule1': 0, 'rule2': 0, 'rule3': 0, 'rule4': 0, 'rule5': 0
        }
        self._daily_realized_r = 0.0
        self._current_day = None
        self._consecutive_losses = 0
        self._cooldown_bars_remaining = 0
        self._equity_history = []
        try:
            self._current_day = str(self.env._current_row()['date_str'])
        except Exception:
            pass
        return obs, info

    def step(self, action: int):
        modified_action = int(action)
        rules_fired: list[str] = []
        self._blocked_reason_val = 'none'

        # ── Get current bar date ─────────────────────────────────────────────
        try:
            date_str = str(self.env._current_row()['date_str'])
        except Exception:
            date_str = self._current_day or '1970-01-01'

        # Reset daily tracker on calendar day change
        if date_str != self._current_day:
            self._current_day = date_str
            self._daily_realized_r = 0.0

        # ── Rule 1: Daily loss limit ─────────────────────────────────────────
        if modified_action in _ENTRY_ACTIONS:
            if self._daily_realized_r <= -3.0:
                modified_action = PASS
                rules_fired.append('rule1')
                self._blocked_reason_val = 'daily_loss_limit'
                self._rules_triggered_val['rule1'] += 1

        # ── Rule 2: Consecutive loss circuit breaker ─────────────────────────
        if modified_action in _ENTRY_ACTIONS:
            if self._cooldown_bars_remaining > 0:
                modified_action = PASS
                rules_fired.append('rule2')
                if self._blocked_reason_val == 'none':
                    self._blocked_reason_val = 'consecutive_loss_circuit_breaker'
                self._rules_triggered_val['rule2'] += 1

        # ── Rule 3: Correlation guard ────────────────────────────────────────
        if modified_action in _ENTRY_ACTIONS:
            try:
                current_pair = str(self.env.current_pair)
                usd_strength = float(np.tanh(self.env._usd_idx.get(date_str, 0.0)))
                jpy_strength = float(np.tanh(self.env._jpy_idx.get(date_str, 0.0)))
                corr_blocked = (
                    (abs(usd_strength) > 0.7 and self._is_usd_pair(current_pair)) or
                    (abs(jpy_strength) > 0.7 and self._is_jpy_pair(current_pair))
                )
            except AttributeError:
                corr_blocked = False
            if corr_blocked:
                modified_action = PASS
                rules_fired.append('rule3')
                if self._blocked_reason_val == 'none':
                    self._blocked_reason_val = 'correlation_guard'
                self._rules_triggered_val['rule3'] += 1

        # ── Rule 4: Maximum bars in trade hard stop ──────────────────────────
        try:
            bars_in_trade = int(self.env._bars_in_trade)
            in_trade = bool(self.env._in_trade)
        except AttributeError:
            bars_in_trade = 0
            in_trade = False

        if in_trade and bars_in_trade > 10 and modified_action != CLOSE:
            modified_action = CLOSE
            rules_fired.append('rule4')
            if self._blocked_reason_val == 'none':
                self._blocked_reason_val = 'max_bars_in_trade'
            self._rules_triggered_val['rule4'] += 1

        # ── Rule 5: Equity curve filter ──────────────────────────────────────
        if modified_action in _ENTRY_ACTIONS and len(self._equity_history) >= 20:
            rolling_avg = float(np.mean(self._equity_history[-20:]))
            try:
                current_equity = float(self.env._equity)
            except AttributeError:
                current_equity = rolling_avg
            if rolling_avg > 0.0 and current_equity < rolling_avg * 0.85:
                modified_action = PASS
                rules_fired.append('rule5')
                if self._blocked_reason_val == 'none':
                    self._blocked_reason_val = 'equity_curve_filter'
                self._rules_triggered_val['rule5'] += 1

        # ── Execute modified action ──────────────────────────────────────────
        penalty = 0.02 * len(rules_fired)
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        reward -= penalty

        # Update equity rolling history
        try:
            self._equity_history.append(float(self.env._equity))
        except AttributeError:
            pass

        # Update guard state from any trade closure this step
        newly_set_cooldown = False
        if 'realized_r' in info:
            r = float(info['realized_r'])
            self._daily_realized_r += r
            if r < 0.0:
                self._consecutive_losses += 1
                if self._consecutive_losses >= 3:
                    self._cooldown_bars_remaining = 5
                    newly_set_cooldown = True
            else:
                # Any winning trade resets the consecutive loss streak
                self._consecutive_losses = 0

        # Tick down the cooldown each bar (but not the bar it was just set)
        if not newly_set_cooldown and self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1

        info['guard_blocked_reason'] = self._blocked_reason_val
        info['rules_triggered'] = dict(self._rules_triggered_val)
        return obs, reward, terminated, truncated, info

    # ── Pair helpers ─────────────────────────────────────────────────────────

    def _is_usd_pair(self, pair: str) -> bool:
        return pair.upper() in _USD_PAIRS

    def _is_jpy_pair(self, pair: str) -> bool:
        return pair.upper() in _JPY_PAIRS

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def blocked_reason(self) -> str:
        """Which rule blocked action on the last step, or 'none'."""
        return self._blocked_reason_val

    @property
    def rules_triggered(self) -> dict:
        """Cumulative count of rule firings this episode."""
        return dict(self._rules_triggered_val)
