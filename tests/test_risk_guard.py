"""
Pytest tests for RiskGuardWrapper (Step 4).
"""

import os
import sys
import numpy as np
import pytest
import gymnasium

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from rl_agent.risk_guard import (
    RiskGuardWrapper,
    PASS, ENTER_LONG, ENTER_SHORT, HOLD, CLOSE, PARTIAL_CLOSE,
)


# ── Minimal mock environment ──────────────────────────────────────────────────

class MockForexEnv(gymnasium.Env):
    """
    Thin stand-in for ForexTradingEnv.  All attributes the wrapper reads are
    exposed so we can manipulate them directly in tests.
    """

    def __init__(self, pair: str = 'EURUSD', date_str: str = '2020-01-01'):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(6)
        self.observation_space = gymnasium.spaces.Box(
            low=-10.0, high=10.0, shape=(66,), dtype=np.float32
        )
        self._current_pair = pair
        self._date_str = date_str
        self._bars_in_trade = 0
        self._in_trade = False
        self._equity = 100_000.0

        # Correlation lookup tables (tanh applied by wrapper)
        self._usd_idx: dict = {}
        self._jpy_idx: dict = {}

        # Configurable step behaviour
        self._step_realized_r: float | None = None
        self._step_reward: float = 0.0
        self.last_action: int | None = None

    @property
    def current_pair(self) -> str:
        return self._current_pair

    def _current_row(self) -> dict:
        return {'date_str': self._date_str}

    def reset(self, seed=None, options=None):
        self._bars_in_trade = 0
        self._in_trade = False
        self._equity = 100_000.0
        self.last_action = None
        return np.zeros(66, dtype=np.float32), {'equity': self._equity}

    def step(self, action: int):
        self.last_action = action
        obs = np.zeros(66, dtype=np.float32)
        reward = self._step_reward
        info: dict = {'equity': self._equity}
        if self._step_realized_r is not None:
            info['realized_r'] = self._step_realized_r
        return obs, reward, False, False, info


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wrapper(pair='EURUSD', date_str='2020-01-01') -> tuple:
    env = MockForexEnv(pair=pair, date_str=date_str)
    wrapper = RiskGuardWrapper(env)
    wrapper.reset()
    return wrapper, env


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_daily_loss_limit_blocks_entry():
    """After -3R realized on the same day, ENTER_LONG converts to PASS."""
    wrapper, env = _make_wrapper(date_str='2020-01-01')

    # Inject -3R accumulated loss without stepping (unit-test internal state)
    wrapper._daily_realized_r = -3.0
    wrapper._current_day = '2020-01-01'

    _, reward, _, _, info = wrapper.step(ENTER_LONG)

    assert env.last_action == PASS, "ENTER_LONG must be converted to PASS"
    assert info['guard_blocked_reason'] == 'daily_loss_limit'
    assert wrapper.rules_triggered['rule1'] == 1
    assert reward == pytest.approx(-0.02)   # one rule fired → 0.02 penalty


def test_consecutive_loss_circuit_breaker():
    """After 3 consecutive losses, entries are blocked for the next 5 bars."""
    wrapper, env = _make_wrapper()

    # Use -0.9R per trade so total = -2.7R, safely below the -3R daily limit
    # (prevents Rule 1 from masking the Rule 2 cooldown after expiry)
    env._step_realized_r = -0.9
    wrapper.step(CLOSE)
    wrapper.step(CLOSE)
    wrapper.step(CLOSE)   # 3rd loss → cooldown = 5 (not decremented this bar)

    # Entries must be blocked for exactly 5 subsequent bars
    env._step_realized_r = None
    for i in range(5):
        wrapper.step(ENTER_LONG)
        assert env.last_action == PASS, f"Bar {i+1} of cooldown should be blocked"

    # 6th bar: cooldown exhausted, entry is allowed
    wrapper.step(ENTER_LONG)
    assert env.last_action == ENTER_LONG, "Entry must be permitted after cooldown"


def test_consecutive_loss_resets_on_win():
    """A winning trade resets the consecutive-loss counter to zero."""
    wrapper, env = _make_wrapper()

    env._step_realized_r = -1.0
    wrapper.step(CLOSE)
    wrapper.step(CLOSE)
    assert wrapper._consecutive_losses == 2

    env._step_realized_r = 1.0   # winning trade
    wrapper.step(CLOSE)

    assert wrapper._consecutive_losses == 0
    assert wrapper._cooldown_bars_remaining == 0

    # Entries should be allowed immediately after the win
    env._step_realized_r = None
    wrapper.step(ENTER_LONG)
    assert env.last_action == ENTER_LONG


def test_correlation_guard_usd():
    """When abs(usd_strength) > 0.7, ENTER_LONG on a USD pair is blocked."""
    wrapper, env = _make_wrapper(pair='EURUSD', date_str='2020-01-01')

    # tanh(1.5) ≈ 0.905, which is > 0.7
    env._usd_idx = {'2020-01-01': 1.5}

    _, _, _, _, info = wrapper.step(ENTER_LONG)

    assert env.last_action == PASS, "ENTER_LONG on USD pair must be blocked"
    assert info['guard_blocked_reason'] == 'correlation_guard'
    assert wrapper.rules_triggered['rule3'] == 1


def test_max_bars_force_close():
    """After > 10 bars in trade, the action is converted to CLOSE regardless."""
    wrapper, env = _make_wrapper()

    env._in_trade = True
    env._bars_in_trade = 11    # 11 > 10 → triggers rule

    _, _, _, _, info = wrapper.step(HOLD)   # agent wants to hold

    assert env.last_action == CLOSE, "Must force-close after > 10 bars in trade"
    assert info['guard_blocked_reason'] == 'max_bars_in_trade'
    assert wrapper.rules_triggered['rule4'] == 1


def test_equity_curve_filter():
    """When equity > 15% below 20-bar rolling average, entries are blocked."""
    wrapper, env = _make_wrapper()

    # Prime 20 bars of equity history at 100k
    wrapper._equity_history = [100_000.0] * 20

    # Drop current equity to 84k (100k * 0.85 = 85k, so 84k is below threshold)
    env._equity = 84_000.0

    _, _, _, _, info = wrapper.step(ENTER_LONG)

    assert env.last_action == PASS, "Entry must be blocked by equity curve filter"
    assert info['guard_blocked_reason'] == 'equity_curve_filter'
    assert wrapper.rules_triggered['rule5'] == 1


def test_blocked_reason_property():
    """blocked_reason property reflects the most recent rule that fired."""
    wrapper, env = _make_wrapper(date_str='2020-01-01')

    assert wrapper.blocked_reason == 'none'

    wrapper._daily_realized_r = -3.0
    wrapper._current_day = '2020-01-01'
    wrapper.step(ENTER_LONG)

    assert wrapper.blocked_reason == 'daily_loss_limit'

    # A step with no rule firing resets reason to 'none'
    wrapper._daily_realized_r = 0.0
    wrapper.step(HOLD)
    assert wrapper.blocked_reason == 'none'


def test_rules_triggered_counter():
    """rules_triggered dict accumulates counts across the episode."""
    wrapper, env = _make_wrapper(date_str='2020-01-01')

    wrapper._daily_realized_r = -3.0
    wrapper._current_day = '2020-01-01'

    wrapper.step(ENTER_LONG)
    wrapper.step(ENTER_LONG)

    triggered = wrapper.rules_triggered
    assert triggered['rule1'] == 2
    assert triggered['rule2'] == 0
    assert triggered['rule3'] == 0
    assert triggered['rule4'] == 0
    assert triggered['rule5'] == 0


def test_no_force_close_existing_on_correlation():
    """Correlation guard only blocks new entries; it never forces-close existing positions."""
    wrapper, env = _make_wrapper(pair='EURUSD', date_str='2020-01-01')

    # High USD strength
    env._usd_idx = {'2020-01-01': 1.5}

    # Existing trade well within bars limit
    env._in_trade = True
    env._bars_in_trade = 2

    # HOLD on an existing trade must NOT be converted to CLOSE
    wrapper.step(HOLD)
    assert env.last_action == HOLD, "Correlation guard must not affect HOLD action"

    # But ENTER_LONG must be blocked
    wrapper.step(ENTER_LONG)
    assert env.last_action == PASS, "ENTER_LONG must be blocked by correlation guard"


def test_wrapper_transparent_spaces():
    """Observation and action spaces pass through unchanged from the wrapped env."""
    env = MockForexEnv()
    wrapper = RiskGuardWrapper(env)

    assert wrapper.action_space == env.action_space
    assert wrapper.observation_space == env.observation_space
    assert wrapper.observation_space.shape == (66,)
    assert wrapper.action_space.n == 6
