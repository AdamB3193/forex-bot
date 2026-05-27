"""
Step 7A: Final Training Run — Nick Shawn Forex Trading Bot
===========================================================
Loads best hyperparameters from models/best_params.json (Step 6 output)
and trains PPO on the full train split (2005-2019) with all 24 pairs.

Saves production model to:
  models/final/final_model.zip        <- loaded by MT5 executor
  models/final/best/best_model.zip    <- best checkpoint during training

Usage:
  python src/rl_agent/final_trainer.py                       # 500K steps
  python src/rl_agent/final_trainer.py --timesteps 1000000   # 1M steps
  python src/rl_agent/final_trainer.py --device cpu          # force CPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data_pipeline.config import DB_PATH, PAIRS
from rl_agent.environment import ForexTradingEnv
from rl_agent.risk_guard import RiskGuardWrapper
from rl_agent.trainer import LSTMFeatureExtractor, make_env

BEST_PARAMS_PATH = ROOT / "models" / "best_params.json"
FINAL_DIR        = ROOT / "models" / "final"
FINAL_MODEL_PATH = FINAL_DIR / "final_model"
BEST_CKPT_DIR    = FINAL_DIR / "best"
EVAL_LOG_DIR     = FINAL_DIR / "eval_logs"


def load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"best_params.json not found at {BEST_PARAMS_PATH}. "
            "Run Step 6 first: python src/rl_agent/hyperopt.py --trials 75"
        )
    with open(BEST_PARAMS_PATH) as f:
        data = json.load(f)
    return data["params"]


def build_policy_kwargs(params: dict) -> dict:
    return dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden=int(params["lstm_hidden"]),
            lstm_layers=int(params["lstm_layers"]),
            fc1_out=int(params["fc1_out"]),
            fc2_out=128,
        ),
        net_arch=[],
        activation_fn=torch.nn.ReLU,
    )


def run_final_training(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    eval_freq: int = 25_000,
    device: str = "auto",
    seed: int = 42,
) -> PPO:
    t0 = time.time()
    params = load_best_params()
    all_pairs = list(PAIRS.keys())

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    W = 62
    print("=" * W)
    print("  STEP 7A: FINAL TRAINING RUN")
    print("-" * W)
    print(f"  Pairs           : {len(all_pairs)} (full pair set)")
    print(f"  Train split     : 2005-01-01 -> 2019-12-31")
    print(f"  Eval split      : 2020-01-01 -> 2022-12-31 (val)")
    print(f"  TEST SEALED     : 2023-01-01 -> present  DO NOT TOUCH")
    print(f"  Timesteps       : {total_timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Device          : {device}")
    print(f"  lstm_hidden     : {params['lstm_hidden']}")
    print(f"  lstm_layers     : {params['lstm_layers']}")
    print(f"  fc1_out         : {params['fc1_out']}")
    print(f"  learning_rate   : {params['learning_rate']:.2e}")
    print(f"  n_steps         : {params['n_steps']}")
    print(f"  batch_size      : {params['batch_size']}")
    print("=" * W)

    train_vec_env = make_vec_env(
        make_env(str(DB_PATH), all_pairs, split="train", use_reward_shaping=True),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = Monitor(RiskGuardWrapper(
        ForexTradingEnv(db_path=str(DB_PATH), pairs=all_pairs, split="val")
    ))

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        best_model_save_path=str(BEST_CKPT_DIR),
        log_path=str(EVAL_LOG_DIR),
        deterministic=True,
        render=False,
        verbose=1,
    )

    policy_kwargs = build_policy_kwargs(params)
    model = PPO(
        "MlpPolicy",
        train_vec_env,
        learning_rate=float(params["learning_rate"]),
        n_steps=int(params["n_steps"]),
        batch_size=int(params["batch_size"]),
        gamma=float(params["gamma"]),
        gae_lambda=float(params["gae_lambda"]),
        clip_range=float(params["clip_range"]),
        n_epochs=int(params["n_epochs"]),
        ent_coef=float(params["ent_coef"]),
        vf_coef=float(params["vf_coef"]),
        max_grad_norm=float(params["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save(str(FINAL_MODEL_PATH))
    elapsed = time.time() - t0

    print("\n" + "=" * W)
    print("  FINAL TRAINING COMPLETE")
    print("-" * W)
    print(f"  Best eval reward: {eval_callback.best_mean_reward:.4f}")
    print(f"  Wall-clock time : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Final model     -> {FINAL_MODEL_PATH}.zip")
    print(f"  Best checkpoint -> {BEST_CKPT_DIR}/best_model.zip")
    print("=" * W + "\n")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7A: Final Training Run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timesteps", type=int,  default=500_000)
    p.add_argument("--envs",      type=int,  default=4)
    p.add_argument("--eval-freq", type=int,  default=25_000)
    p.add_argument("--device",    type=str,  default="auto")
    p.add_argument("--seed",      type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_final_training(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        device=args.device,
        seed=args.seed,
    )
