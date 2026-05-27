"""
Pytest tests for trainer.py (Step 5).
"""

import os
import sys
import tempfile

import numpy as np
import gymnasium
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from data_pipeline.config import DB_PATH, PAIRS
from rl_agent.trainer import LSTMFeatureExtractor, make_env, build_model, run_training
from rl_agent.environment import ForexTradingEnv
from rl_agent.risk_guard import RiskGuardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

_PAIRS = list(PAIRS.keys())[:6]
_WINDOW = 60
_TMPDIR = tempfile.mkdtemp()


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def obs_space():
    return gymnasium.spaces.Box(low=-10.0, high=10.0, shape=(66,), dtype=np.float32)


@pytest.fixture(scope='module')
def extractor(obs_space):
    return LSTMFeatureExtractor(obs_space)


@pytest.fixture(scope='module')
def model():
    vec_env = sb3_make_vec_env(
        make_env(DB_PATH, _PAIRS, split='train', window_size=_WINDOW),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    return build_model(vec_env, device='cpu')


@pytest.fixture(scope='module')
def trained_result():
    return run_training(
        db_path=DB_PATH,
        pairs=_PAIRS,
        total_timesteps=2048,
        n_envs=1,
        eval_freq=2048,
        save_dir=os.path.join(_TMPDIR, 'models'),
        run_name='test_run',
        device='cpu',
        seed=0,
    )


# ── 1. LSTMFeatureExtractor output shape ────────────────────────────────────

def test_lstm_extractor_output_shape(extractor):
    obs = torch.zeros(4, 66)
    out = extractor(obs)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"


# ── 2. LSTMFeatureExtractor output dtype ────────────────────────────────────

def test_lstm_extractor_dtype(extractor):
    obs = torch.zeros(4, 66)
    out = extractor(obs)
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


# ── 3. build_model returns PPO ───────────────────────────────────────────────

def test_build_model_returns_ppo(model):
    assert isinstance(model, PPO)


# ── 4. build_model has LSTMFeatureExtractor ──────────────────────────────────

def test_build_model_has_lstm_extractor(model):
    assert isinstance(model.policy.features_extractor, LSTMFeatureExtractor)


# ── 5. make_env returns callable that returns RiskGuardWrapper ───────────────

def test_make_env_callable():
    thunk = make_env(DB_PATH, _PAIRS, split='train', window_size=_WINDOW)
    assert callable(thunk)
    env = thunk()
    inner = env.env if hasattr(env, 'env') else env
    assert isinstance(inner, RiskGuardWrapper)
    env.close()


# ── 6. model.predict returns action in [0, 5] ───────────────────────────────

def test_model_predict_shape(model):
    obs = np.zeros(66, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action.flat[0]) <= 5, f"Action {action} out of [0, 5]"


# ── 7. model.predict is deterministic ────────────────────────────────────────

def test_model_predict_deterministic(model):
    obs = np.zeros(66, dtype=np.float32)
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = model.predict(obs, deterministic=True)
    assert int(a1.flat[0]) == int(a2.flat[0])


# ── 8. model save/load round-trip ────────────────────────────────────────────

def test_model_saves_and_loads(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'roundtrip_model')
        model.save(path)
        loaded = PPO.load(path, device='cpu')
        obs = np.zeros(66, dtype=np.float32)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        assert int(a1.flat[0]) == int(a2.flat[0])


# ── 9. run_training short run completes and returns PPO ──────────────────────

def test_run_training_short(trained_result):
    assert isinstance(trained_result, PPO)


# ── 10. EvalCallback creates best_model directory ────────────────────────────

def test_eval_callback_creates_files(trained_result):
    best_dir = os.path.join(_TMPDIR, 'models', 'test_run', 'best_model')
    assert os.path.isdir(best_dir), f"best_model directory not found at {best_dir}"
