"""
PPO training module for the forex RL agent.
"""

import os
import sys
import time

import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_agent.environment import ForexTradingEnv
from rl_agent.risk_guard import RiskGuardWrapper


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Stateless LSTM feature extractor for PPO MlpPolicy.

    Each forward pass treats the single observation as a sequence of length 1,
    resetting hidden state to zeros each call. This gives the network LSTM
    capacity without requiring sb3-contrib's RecurrentPPO.
    """

    def __init__(self, observation_space, lstm_hidden=128, lstm_layers=2,
                 fc1_out=256, fc2_out=128):
        super().__init__(observation_space, features_dim=fc2_out)
        input_size = observation_space.shape[0]
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, fc1_out),
            torch.nn.ReLU(),
            torch.nn.Linear(fc1_out, fc2_out),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, 66) → unsqueeze to (batch, 1, 66)
        x = observations.unsqueeze(1)
        h0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        c0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        out, _ = self.lstm(x, (h0, c0))   # (batch, 1, lstm_hidden)
        out = out[:, -1, :]               # (batch, lstm_hidden)
        return self.fc(out)               # (batch, fc2_out=128)


def make_env(db_path, pairs, split='train', window_size=252, use_reward_shaping=False):
    def _init():
        env = ForexTradingEnv(
            db_path=db_path, pairs=pairs, split=split,
            window_size=window_size, use_reward_shaping=use_reward_shaping
        )
        return Monitor(RiskGuardWrapper(env))
    return _init


def build_model(train_env, device='auto') -> PPO:
    """Build a PPO model with the custom LSTM feature extractor."""
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden=128, lstm_layers=2, fc1_out=256, fc2_out=128,
        ),
        net_arch=[],
        activation_fn=torch.nn.ReLU,
    )
    return PPO(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        verbose=1,
        device=device,
    )


def run_training(
    db_path: str,
    pairs: list,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    eval_freq: int = 10_000,
    save_dir: str = 'models',
    run_name: str = 'ppo_forex_v1',
    device: str = 'auto',
    seed: int = 42,
) -> PPO:
    t0 = time.time()

    # 1. Vectorized training environment (DummyVecEnv avoids Windows multiprocessing issues)
    train_vec_env = make_vec_env(
        make_env(db_path, pairs, split='train'),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    # 2. Single validation environment for evaluation (never trained on)
    eval_env = Monitor(RiskGuardWrapper(
        ForexTradingEnv(db_path=db_path, pairs=pairs, split='val')
    ))

    # 3. EvalCallback — evaluates on val split, saves best model
    best_model_path = os.path.join(save_dir, run_name, 'best_model')
    log_path = os.path.join(save_dir, run_name, 'eval_logs')
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        best_model_save_path=best_model_path,
        log_path=log_path,
        deterministic=True,
        render=False,
    )

    # 4. Build PPO model
    model = build_model(train_vec_env, device=device)

    # 5. Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # 6. Save final model
    final_dir = os.path.join(save_dir, run_name)
    os.makedirs(final_dir, exist_ok=True)
    model.save(os.path.join(final_dir, 'final_model'))

    # 7. Training summary
    elapsed = time.time() - t0
    best_reward = eval_callback.best_mean_reward
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Best eval reward: {best_reward:.4f}")
    print(f"  Models saved to : {final_dir}/")
    print(f"  Wall-clock time : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train forex PPO agent')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--envs',      type=int, default=4)
    parser.add_argument('--eval-freq', type=int, default=10_000)
    parser.add_argument('--run-name',  type=str, default='ppo_forex_v1')
    parser.add_argument('--device',    type=str, default='auto')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.config import DB_PATH, PAIRS

    run_training(
        db_path=DB_PATH,
        pairs=list(PAIRS.keys()),
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
    )
