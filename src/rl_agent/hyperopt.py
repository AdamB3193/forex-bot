"""
Step 6: Hyperparameter Optimization — Nick Shawn Forex Trading Bot
==================================================================
Uses Optuna (TPE sampler + MedianPruner) to sweep:
  - PPO hyperparameters: learning_rate, n_steps, batch_size, gamma,
    gae_lambda, clip_range, n_epochs, ent_coef, vf_coef, max_grad_norm
  - LSTM architecture: lstm_hidden, lstm_layers, fc1_out
    (names match LSTMFeatureExtractor.__init__ exactly)

Objective: mean walk-forward OOS Sharpe ratio across 5 folds.
           All folds stay within the 2020-2022 validation period.
           TEST SET (2023-2026) SEALED — never touched.

Walk-forward mechanics:
  Each fold trains PPO on split='train' (2005-2019) for TIMESTEPS_PER_FOLD
  steps, then evaluates on a specific OOS window within split='val' (2020-2022).
  Requires the date_start / date_end patch to be applied to environment.py.
  The eval falls back to the full val split if the patch is not yet applied.

Persistence: Study stored in data/db/optuna_study.db (crash-safe / resumable).
Best params saved to models/best_params.json.

CLI:
  python src/rl_agent/hyperopt.py --trials 75
  python src/rl_agent/hyperopt.py --trials 100 --resume
  python src/rl_agent/hyperopt.py --show-best
  python src/rl_agent/hyperopt.py --trials 5 --timesteps-per-fold 5000
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data_pipeline.config import DB_PATH, PAIRS
from rl_agent.environment import ForexTradingEnv
from rl_agent.risk_guard import RiskGuardWrapper
from rl_agent.trainer import LSTMFeatureExtractor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_DB_PATH   = ROOT / "data" / "db" / "optuna_study.db"
MODELS_DIR       = ROOT / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
MODELS_DIR.mkdir(exist_ok=True)

_ENV_SUPPORTS_DATES: bool = (
    "date_start" in inspect.signature(ForexTradingEnv.__init__).parameters
)

_CANDIDATE_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY",
]
HYPEROPT_PAIRS: List[str] = [p for p in _CANDIDATE_PAIRS if p in PAIRS]

WALK_FORWARD_WINDOWS: List[Dict[str, str]] = [
    {"name": "WF-1", "eval_start": "2020-01-01", "eval_end": "2020-06-30"},
    {"name": "WF-2", "eval_start": "2020-07-01", "eval_end": "2020-12-31"},
    {"name": "WF-3", "eval_start": "2021-01-01", "eval_end": "2021-06-30"},
    {"name": "WF-4", "eval_start": "2021-07-01", "eval_end": "2021-12-31"},
    {"name": "WF-5", "eval_start": "2022-01-01", "eval_end": "2022-12-31"},
]

TIMESTEPS_PER_FOLD: int = 60_000
N_EVAL_EPISODES:    int = 5
WINDOW_SIZE:        int = 252
PRUNING_FREQ:       int = 10_000


def _make_train_thunk(seed: int = 0):
    def _init() -> Monitor:
        env = ForexTradingEnv(
            db_path=str(DB_PATH),
            pairs=HYPEROPT_PAIRS,
            split="train",
            window_size=WINDOW_SIZE,
        )
        mon = Monitor(RiskGuardWrapper(env))
        mon.reset(seed=seed)
        return mon
    return _init


def _make_eval_env(date_start: str, date_end: str, seed: int = 0) -> Monitor:
    if _ENV_SUPPORTS_DATES:
        env = ForexTradingEnv(
            db_path=str(DB_PATH),
            pairs=HYPEROPT_PAIRS,
            split="val",
            window_size=min(WINDOW_SIZE, 125),
            date_start=date_start,
            date_end=date_end,
        )
    else:
        logger.warning(
            "ForexTradingEnv does not support date_start/date_end — "
            "apply environment.py patch for true walk-forward OOS. "
            "Falling back to full val split."
        )
        env = ForexTradingEnv(
            db_path=str(DB_PATH),
            pairs=HYPEROPT_PAIRS,
            split="val",
            window_size=WINDOW_SIZE,
        )
    mon = Monitor(RiskGuardWrapper(env))
    mon.reset(seed=seed + 1000)
    return mon


def _build_policy_kwargs(
    lstm_hidden: int,
    lstm_layers: int,
    fc1_out: int,
    fc2_out: int = 128,
) -> dict:
    return dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            fc1_out=fc1_out,
            fc2_out=fc2_out,
        ),
        net_arch=[],
        activation_fn=torch.nn.ReLU,
    )


class TrialPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, fold_offset: int = 0):
        super().__init__(verbose=0)
        self.trial       = trial
        self.fold_offset = fold_offset
        self._local_step = 0

    def _on_step(self) -> bool:
        self._local_step += 1
        if self._local_step % PRUNING_FREQ != 0:
            return True
        mean_reward = -999.0
        if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
            rews = [ep["r"] for ep in self.model.ep_info_buffer]
            if rews:
                mean_reward = float(np.mean(rews[-20:]))
        self.trial.report(mean_reward, step=self.fold_offset + self._local_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def _evaluate_sharpe(
    model: PPO,
    date_start: str,
    date_end: str,
    n_episodes: int = N_EVAL_EPISODES,
) -> float:
    episode_returns: List[float] = []
    for ep_idx in range(n_episodes):
        try:
            env = _make_eval_env(date_start, date_end, seed=ep_idx)
            obs, _ = env.reset(seed=ep_idx + 1000)
            done = False
            ep_return = 0.0
            step_count = 0
            while not done and step_count < WINDOW_SIZE * 3:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                ep_return  += float(reward)
                done        = terminated or truncated
                step_count += 1
            episode_returns.append(ep_return)
        except Exception as exc:
            logger.debug(f"  Eval ep {ep_idx} error: {exc}")
            episode_returns.append(-10.0)
    if not episode_returns:
        return -999.0
    if len(episode_returns) == 1:
        return float(episode_returns[0])
    arr = np.array(episode_returns, dtype=np.float64)
    return float(arr.mean() / (arr.std() + 1e-8))


def _run_walk_forward(trial: optuna.Trial, params: Dict[str, Any]) -> float:
    fold_sharpes: List[float] = []
    for fold_idx, window in enumerate(WALK_FORWARD_WINDOWS):
        wf_name = window["name"]
        logger.info(
            f"  [Trial {trial.number:>3}] {wf_name}: "
            f"train=2005-2019 | eval={window['eval_start']}→{window['eval_end']}"
        )
        train_env = DummyVecEnv([_make_train_thunk(seed=fold_idx * 17)])
        policy_kwargs = _build_policy_kwargs(
            lstm_hidden=params["lstm_hidden"],
            lstm_layers=params["lstm_layers"],
            fc1_out=params["fc1_out"],
            fc2_out=128,
        )
        try:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                n_epochs=params["n_epochs"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                policy_kwargs=policy_kwargs,
                verbose=0,
                seed=fold_idx * 31 + trial.number,
                device="cpu",
            )
        except Exception as exc:
            logger.warning(f"  {wf_name}: model build failed — {exc}")
            train_env.close()
            return -999.0
        cb = TrialPruningCallback(trial, fold_offset=fold_idx * TIMESTEPS_PER_FOLD)
        try:
            model.learn(
                total_timesteps=TIMESTEPS_PER_FOLD,
                callback=cb,
                reset_num_timesteps=True,
                progress_bar=False,
            )
        except optuna.TrialPruned:
            train_env.close()
            raise
        except Exception as exc:
            logger.warning(f"  {wf_name}: training crashed — {exc}")
            train_env.close()
            fold_sharpes.append(-10.0)
            continue
        train_env.close()
        sharpe = _evaluate_sharpe(
            model,
            date_start=window["eval_start"],
            date_end=window["eval_end"],
        )
        logger.info(f"  {wf_name}: OOS Sharpe = {sharpe:+.4f}")
        fold_sharpes.append(sharpe)
        trial.report(
            float(np.mean(fold_sharpes)),
            step=(fold_idx + 1) * TIMESTEPS_PER_FOLD,
        )
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_sharpes)) if fold_sharpes else -999.0


def objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    n_steps       = trial.suggest_categorical("n_steps",   [512, 1024, 2048])
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma         = trial.suggest_float("gamma",           0.95,  0.9999)
    gae_lambda    = trial.suggest_float("gae_lambda",      0.90,  0.99)
    clip_range    = trial.suggest_float("clip_range",      0.10,  0.30)
    n_epochs      = trial.suggest_categorical("n_epochs",  [3, 5, 10])
    ent_coef      = trial.suggest_float("ent_coef",        1e-4,  0.01, log=True)
    vf_coef       = trial.suggest_float("vf_coef",         0.25,  1.00)
    max_grad_norm = trial.suggest_float("max_grad_norm",   0.30,  1.00)
    lstm_hidden   = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
    lstm_layers   = trial.suggest_categorical("lstm_layers", [1, 2])
    fc1_out       = trial.suggest_categorical("fc1_out",     [128, 256, 512])

    if batch_size > n_steps:
        raise optuna.TrialPruned(f"batch_size ({batch_size}) > n_steps ({n_steps})")

    params: Dict[str, Any] = dict(
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
        n_epochs=n_epochs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers, fc1_out=fc1_out,
    )
    logger.info(
        f"Trial {trial.number:>3} | lr={learning_rate:.2e}  n_steps={n_steps}  "
        f"batch={batch_size}  epochs={n_epochs}  "
        f"lstm={lstm_hidden}h×{lstm_layers}L  fc1={fc1_out}"
    )
    return _run_walk_forward(trial, params)


def save_best_params(study: optuna.Study) -> None:
    best     = study.best_trial
    all_t    = study.trials
    complete = [t for t in all_t if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in all_t if t.state == optuna.trial.TrialState.PRUNED]
    top5     = sorted(complete, key=lambda t: t.value, reverse=True)[:5]
    out = {
        "trial_number":       best.number,
        "oos_sharpe":         round(best.value, 6),
        "params":             best.params,
        "n_trials_total":     len(all_t),
        "n_trials_completed": len(complete),
        "n_trials_pruned":    len(pruned),
        "top_5_trials": [
            {"trial": t.number, "oos_sharpe": round(t.value, 6), "params": t.params}
            for t in top5
        ],
        "optimization_history": [
            {"trial": t.number, "value": round(t.value, 6)}
            for t in complete
        ],
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Best params saved → {BEST_PARAMS_PATH}")


def load_best_params() -> Optional[Dict[str, Any]]:
    if not BEST_PARAMS_PATH.exists():
        return None
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


def _print_summary(study: optuna.Study) -> None:
    all_t    = study.trials
    complete = [t for t in all_t if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in all_t if t.state == optuna.trial.TrialState.PRUNED]
    best     = study.best_trial
    W = 62
    print("\n" + "═" * W)
    print("  STEP 6 — HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("═" * W)
    print(f"  Trials total    : {len(all_t)}")
    print(f"  Completed       : {len(complete)}")
    print(f"  Pruned          : {len(pruned)}")
    print(f"  Best trial #    : {best.number}")
    print(f"  Best OOS Sharpe : {best.value:+.4f}")
    print()
    print(f"  {'Parameter':<22}  {'Value':>14}")
    print(f"  {'─'*22}  {'─'*14}")
    for k, v in sorted(best.params.items()):
        print(f"  {k:<22}  {v:>14.6f}" if isinstance(v, float)
              else f"  {k:<22}  {v:>14}")
    print()
    print(f"  Saved → {BEST_PARAMS_PATH.relative_to(ROOT)}")
    print("═" * W + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 6: Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trials",             type=int,  default=75)
    p.add_argument("--jobs",               type=int,  default=1)
    p.add_argument("--study-name",         type=str,  default="forex_bot_step6")
    p.add_argument("--timeout",            type=int,  default=None)
    p.add_argument("--timesteps-per-fold", type=int,  default=TIMESTEPS_PER_FOLD)
    p.add_argument("--resume",             action="store_true")
    p.add_argument("--show-best",          action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_best:
        data = load_best_params()
        if data is None:
            print(f"No saved params at {BEST_PARAMS_PATH}")
            sys.exit(1)
        print(json.dumps(data, indent=2))
        sys.exit(0)

    global TIMESTEPS_PER_FOLD
    TIMESTEPS_PER_FOLD = args.timesteps_per_fold

    W = 62
    print("═" * W)
    print("  STEP 6: HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("─" * W)
    print(f"  Trials          : {args.trials}")
    print(f"  Jobs            : {args.jobs}")
    print(f"  Study DB        : {OPTUNA_DB_PATH.relative_to(ROOT)}")
    print(f"  Folds           : {len(WALK_FORWARD_WINDOWS)}")
    print(f"  Pairs           : {len(HYPEROPT_PAIRS)}")
    print(f"  Timesteps/fold  : {TIMESTEPS_PER_FOLD:,}")
    print(f"  Date filter     : {'YES — true walk-forward' if _ENV_SUPPORTS_DATES else 'NO — apply environment.py patch first'}")
    print(f"  TEST SEALED     : 2023-01-01 → present  ⚠  NEVER TOUCHED")
    print("═" * W)

    storage = f"sqlite:///{OPTUNA_DB_PATH}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=TPESampler(n_startup_trials=10, seed=42, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        direction="maximize",
        load_if_exists=True,
    )

    existing = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    if existing:
        logger.info(f"Resumed — {existing} trials already complete.")

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=args.jobs,
            timeout=args.timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C).")

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        logger.error("No completed trials. Check DB_PATH and data pipeline.")
        sys.exit(1)

    save_best_params(study)
    _print_summary(study)


if __name__ == "__main__":
    main()
