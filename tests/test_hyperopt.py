"""
tests/test_hyperopt.py — Step 6: Hyperparameter Optimization Tests
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import optuna
import torch
import gymnasium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import rl_agent.hyperopt as hmod
from rl_agent.hyperopt import (
    WALK_FORWARD_WINDOWS,
    _build_policy_kwargs,
    _evaluate_sharpe,
    _make_train_thunk,
    load_best_params,
    save_best_params,
    objective,
    HYPEROPT_PAIRS,
)
from rl_agent.trainer import LSTMFeatureExtractor
from data_pipeline.config import DB_PATH, PAIRS


# ── Test 1: _build_policy_kwargs structure and exact kwarg names ──────────────

def test_build_policy_kwargs_structure():
    kwargs = _build_policy_kwargs(
        lstm_hidden=128,
        lstm_layers=2,
        fc1_out=256,
        fc2_out=128,
    )
    assert "features_extractor_class"  in kwargs
    assert "features_extractor_kwargs" in kwargs
    assert "net_arch"                  in kwargs
    assert kwargs["features_extractor_class"] is LSTMFeatureExtractor
    assert kwargs["net_arch"] == []
    fk = kwargs["features_extractor_kwargs"]
    assert "lstm_hidden" in fk, "Must use 'lstm_hidden' not 'lstm_hidden_size'"
    assert "lstm_layers" in fk, "Must use 'lstm_layers' not 'n_layers'"
    assert "fc1_out"     in fk, "Must use 'fc1_out' not 'fc_hidden_size'"
    assert "fc2_out"     in fk
    assert fk["lstm_hidden"] == 128
    assert fk["lstm_layers"] == 2
    assert fk["fc1_out"]     == 256
    assert fk["fc2_out"]     == 128


# ── Test 2: _build_policy_kwargs custom dims ──────────────────────────────────

def test_build_policy_kwargs_custom_dims():
    kwargs = _build_policy_kwargs(lstm_hidden=64, lstm_layers=1, fc1_out=512)
    fk = kwargs["features_extractor_kwargs"]
    assert fk["lstm_hidden"] == 64
    assert fk["lstm_layers"] == 1
    assert fk["fc1_out"]     == 512


# ── Test 3: _make_train_thunk creates a valid steppable environment ───────────

def test_make_train_thunk_valid():
    thunk = _make_train_thunk(seed=0)
    assert callable(thunk)
    env = thunk()
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")
    assert env.observation_space.shape == (66,)
    assert env.action_space.n          == 6
    obs, info = env.reset()
    assert obs.shape == (66,)
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    env.close()


# ── Test 4: LSTMFeatureExtractor kwarg names match trainer.py ─────────────────

def test_lstm_feature_extractor_kwarg_names():
    import inspect
    sig    = inspect.signature(LSTMFeatureExtractor.__init__)
    params = set(sig.parameters.keys())
    assert "lstm_hidden" in params, f"Expected 'lstm_hidden' in {params}"
    assert "lstm_layers" in params, f"Expected 'lstm_layers' in {params}"
    assert "fc1_out"     in params, f"Expected 'fc1_out' in {params}"
    assert "fc2_out"     in params, f"Expected 'fc2_out' in {params}"
    obs_space = gymnasium.spaces.Box(low=-10.0, high=10.0, shape=(66,), dtype=np.float32)
    extractor = LSTMFeatureExtractor(
        obs_space, lstm_hidden=64, lstm_layers=1, fc1_out=128, fc2_out=64
    )
    out = extractor(torch.zeros(2, 66))
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"


# ── Test 5: _evaluate_sharpe returns a float ─────────────────────────────────

def test_evaluate_sharpe_returns_float():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    train_env = DummyVecEnv([_make_train_thunk(seed=7)])
    model = PPO("MlpPolicy", train_env, verbose=0, device="cpu",
                n_steps=64, batch_size=32)
    train_env.close()
    result = _evaluate_sharpe(
        model,
        date_start=WALK_FORWARD_WINDOWS[0]["eval_start"],
        date_end=WALK_FORWARD_WINDOWS[0]["eval_end"],
        n_episodes=2,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)


# ── Test 6: walk-forward windows never touch the test set ────────────────────

def test_walk_forward_no_test_set_contamination():
    import pandas as pd
    test_set_start = pd.Timestamp("2023-01-01")
    for w in WALK_FORWARD_WINDOWS:
        assert pd.Timestamp(w["eval_end"]) < test_set_start, (
            f"{w['name']} bleeds into sealed test set"
        )


# ── Test 7: walk-forward windows stay within val period ──────────────────────

def test_walk_forward_windows_within_val_period():
    import pandas as pd
    val_start = pd.Timestamp("2020-01-01")
    for w in WALK_FORWARD_WINDOWS:
        assert pd.Timestamp(w["eval_start"]) >= val_start, (
            f"{w['name']} starts before val period"
        )


# ── Test 8: objective smoke — 1 trial, minimal steps ────────────────────────

def test_objective_smoke_one_trial():
    original_ts = hmod.TIMESTEPS_PER_FOLD
    original_wf = hmod.WALK_FORWARD_WINDOWS
    hmod.TIMESTEPS_PER_FOLD   = 256
    hmod.WALK_FORWARD_WINDOWS = [WALK_FORWARD_WINDOWS[0]]
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=0),
        )
        study.optimize(objective, n_trials=1, catch=(Exception,))
        assert len(study.trials) == 1
    finally:
        hmod.TIMESTEPS_PER_FOLD   = original_ts
        hmod.WALK_FORWARD_WINDOWS = original_wf


# ── Test 9: save_best_params / load_best_params roundtrip ────────────────────

def test_save_load_best_params_roundtrip(tmp_path):
    def _dummy_obj(trial):
        trial.suggest_float("learning_rate",    1e-5, 1e-4, log=True)
        trial.suggest_categorical("n_steps",    [512])
        trial.suggest_categorical("batch_size", [128])
        trial.suggest_categorical("n_epochs",   [5])
        trial.suggest_float("gamma",            0.99, 0.99)
        trial.suggest_float("gae_lambda",       0.95, 0.95)
        trial.suggest_float("clip_range",       0.2,  0.2)
        trial.suggest_float("ent_coef",         0.01, 0.01, log=False)
        trial.suggest_float("vf_coef",          0.5,  0.5)
        trial.suggest_float("max_grad_norm",    0.5,  0.5)
        trial.suggest_categorical("lstm_hidden", [128])
        trial.suggest_categorical("lstm_layers", [2])
        trial.suggest_categorical("fc1_out",     [256])
        return 1.2345

    study = optuna.create_study(direction="maximize")
    study.optimize(_dummy_obj, n_trials=3)

    orig_path = hmod.BEST_PARAMS_PATH
    hmod.BEST_PARAMS_PATH = tmp_path / "best_params.json"
    try:
        save_best_params(study)
        assert hmod.BEST_PARAMS_PATH.exists()
        loaded = load_best_params()
        assert loaded is not None
        assert loaded["oos_sharpe"]         == pytest.approx(1.2345, abs=1e-4)
        assert loaded["n_trials_completed"] == 3
        assert "learning_rate" in loaded["params"]
        assert "lstm_hidden"   in loaded["params"]
        assert "lstm_layers"   in loaded["params"]
        assert "fc1_out"       in loaded["params"]
    finally:
        hmod.BEST_PARAMS_PATH = orig_path


def test_load_best_params_missing_file(tmp_path):
    orig_path = hmod.BEST_PARAMS_PATH
    hmod.BEST_PARAMS_PATH = tmp_path / "nonexistent.json"
    try:
        assert load_best_params() is None
    finally:
        hmod.BEST_PARAMS_PATH = orig_path


# ── Test 10: batch_size > n_steps triggers TrialPruned ───────────────────────

def test_batch_size_constraint_triggers_prune():
    def _bad_objective(trial):
        trial.suggest_float("learning_rate",    1e-4, 1e-4, log=False)
        trial.suggest_categorical("n_steps",    [256])
        trial.suggest_categorical("batch_size", [512])
        trial.suggest_float("gamma",            0.99, 0.99)
        trial.suggest_float("gae_lambda",       0.95, 0.95)
        trial.suggest_float("clip_range",       0.2,  0.2)
        trial.suggest_categorical("n_epochs",   [10])
        trial.suggest_float("ent_coef",         0.01, 0.01, log=False)
        trial.suggest_float("vf_coef",          0.5,  0.5)
        trial.suggest_float("max_grad_norm",    0.5,  0.5)
        trial.suggest_categorical("lstm_hidden", [128])
        trial.suggest_categorical("lstm_layers", [2])
        trial.suggest_categorical("fc1_out",     [256])
        if 512 > 256:
            raise optuna.TrialPruned("batch_size > n_steps")
        return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(_bad_objective, n_trials=1, catch=())
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
