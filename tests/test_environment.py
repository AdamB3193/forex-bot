"""
Pytest tests for ForexTradingEnv (Step 2).
"""

import os
import sys
import numpy as np
import pytest

# Resolve paths so imports work from any cwd
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'data_pipeline'))

from data_pipeline.config import DB_PATH, PAIRS
from rl_agent.environment import ForexTradingEnv, ENTER_LONG, ENTER_SHORT, CLOSE, PASS

# Use a small pair subset + short window so tests run fast
_PAIRS = list(PAIRS.keys())[:6]
_WINDOW = 60


@pytest.fixture(scope='module')
def env():
    e = ForexTradingEnv(db_path=DB_PATH, pairs=_PAIRS, split='train',
                        window_size=_WINDOW, initial_equity=100_000.0)
    yield e


# ── 1. reset() returns correct shape/dtype ───────────────────────────────────

def test_env_reset(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (66,), f"Expected shape (66,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert isinstance(info, dict)


# ── 2. step(PASS) returns valid tuple ────────────────────────────────────────

def test_env_step_pass(env):
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(PASS)
    assert obs.shape == (66,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ── 3. Observation bounds after 50 random steps ──────────────────────────────

def test_obs_bounds(env):
    obs, _ = env.reset(seed=2)
    assert obs.min() >= -10.0 and obs.max() <= 10.0, \
        f"Initial obs out of bounds: min={obs.min()}, max={obs.max()}"
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.min() >= -10.0, f"obs min {obs.min()} < -10"
        assert obs.max() <= 10.0,  f"obs max {obs.max()} > 10"
        if terminated or truncated:
            obs, _ = env.reset()


# ── 4. CLOSE when flat → reward = -0.05 ──────────────────────────────────────

def test_invalid_action_close_when_flat(env):
    env.reset(seed=3)
    # Ensure no trade is open
    assert not env._in_trade
    _, reward, _, _, _ = env.step(CLOSE)
    assert reward == pytest.approx(-0.05, abs=1e-6)


# ── 5. ENTER_LONG when already long → reward = -0.05 ─────────────────────────

def test_invalid_action_enter_when_in_trade(env):
    env.reset(seed=4)
    # Open a trade
    env.step(ENTER_LONG)
    assert env._in_trade, "Expected trade to be open after ENTER_LONG"
    _, reward, _, _, _ = env.step(ENTER_LONG)
    assert reward == pytest.approx(-0.05, abs=1e-6)


# ── 6. Episode terminates on equity blowup ───────────────────────────────────

def test_episode_terminates(env):
    obs, _ = env.reset(seed=5)
    # Force equity below 50% threshold
    env._equity = env.initial_equity * 0.49
    _, _, terminated, _, _ = env.step(PASS)
    assert terminated, "Episode should terminate when equity < 50% of start"


# ── 7. Spread deducted on trade entry ────────────────────────────────────────

def test_spread_deducted_on_entry(env):
    env.reset(seed=6)
    equity_before = env._equity
    env.step(ENTER_LONG)
    # After entry, equity must be strictly lower (spread was deducted)
    assert env._equity < equity_before, \
        f"Equity should drop on entry. Before={equity_before}, after={env._equity}"


# ── 8. Action space has 6 discrete actions ───────────────────────────────────

def test_action_space(env):
    import gymnasium
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)
    assert env.action_space.n == 6


# ── 9. Observation space shape and bounds ────────────────────────────────────

def test_observation_space(env):
    import gymnasium
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    assert env.observation_space.shape == (66,)
    assert (env.observation_space.low  == -10.0).all()
    assert (env.observation_space.high ==  10.0).all()


# ── 10. Same seed → same first observation ───────────────────────────────────

def test_reproducible_reset(env):
    obs_a, _ = env.reset(seed=99)
    obs_b, _ = env.reset(seed=99)
    np.testing.assert_array_equal(obs_a, obs_b,
        err_msg="Same seed should produce identical first observations")
