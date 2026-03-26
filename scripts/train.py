"""
train.py — Main training script for robotics RL
Works on Mac (MPS) and Colab (CUDA) automatically.
Saves checkpoints to Google Drive when available.

Usage:
  Mac:   python scripts/train.py --env HalfCheetah-v4 --total-steps 50000
  Colab: !python scripts/train.py --env HalfCheetah-v4 --total-steps 1000000
"""

import os
import time
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)


# ── Device detection ──────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        print("🖥  CUDA GPU detected — using CUDA")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎  Apple Silicon detected — using MPS")
        return "mps"
    else:
        print("💻  No GPU — using CPU")
        return "cpu"


# ── Save path (Drive on Colab, ~/rl_research on Mac) ─────────────────────────
def get_save_dir(env_id: str, run_name: str) -> str:
    gdrive = "/content/drive/MyDrive/rl_research"
    local  = os.path.expanduser("~/rl_research")
    base   = gdrive if os.path.exists("/content/drive") else local
    path   = os.path.join(base, env_id, run_name)
    os.makedirs(path, exist_ok=True)
    print(f"📁  Saving to: {path}")
    return path


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device   = get_device()
    save_dir = get_save_dir(args.env, args.run_name)

    # Vectorized + normalized envs
    env      = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(args.env, n_envs=1,           seed=args.seed + 100)
    env      = VecNormalize(env,      norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    # Model
    AlgoCls = SAC if args.algo == "sac" else TD3
    model = AlgoCls(
        "MlpPolicy", env,
        learning_rate   = args.lr,
        buffer_size     = args.buffer_size,
        batch_size      = args.batch_size,
        gamma           = 0.99,
        tau             = 0.005,
        verbose         = 1,
        seed            = args.seed,
        device          = device,
        tensorboard_log = os.path.join(save_dir, "tb_logs"),
    )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(save_dir, "best_model"),
        log_path             = os.path.join(save_dir, "eval_logs"),
        eval_freq            = max(10_000 // args.n_envs, 1),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = max(50_000 // args.n_envs, 1),
        save_path   = os.path.join(save_dir, "checkpoints"),
        name_prefix = f"{args.algo}_{args.env}",
    )

    print(f"\n🚀  Training {args.algo.upper()} on {args.env}")
    print(f"    Steps: {args.total_steps:,} | Envs: {args.n_envs} | Device: {device}\n")

    t0 = time.time()
    model.learn(
        total_timesteps = args.total_steps,
        callback        = CallbackList([eval_cb, ckpt_cb]),
        progress_bar    = True,
    )
    elapsed = (time.time() - t0) / 3600
    print(f"\n✅  Done in {elapsed:.2f} hrs")

    # Save final model + obs normalizer
    model.save(os.path.join(save_dir, f"{args.algo}_final"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"💾  All files saved to {save_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a robotics RL agent")
    p.add_argument("--env",          default="HalfCheetah-v4",
                   help="Gymnasium environment ID")
    p.add_argument("--algo",         default="sac", choices=["sac", "td3"])
    p.add_argument("--run-name",     default="run_01",
                   help="Name for this run (used in save path)")
    p.add_argument("--total-steps",  type=int, default=1_000_000)
    p.add_argument("--n-envs",       type=int, default=4,
                   help="Parallel envs (use 1 for debugging)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--buffer-size",  type=int, default=1_000_000)
    p.add_argument("--batch-size",   type=int, default=256)
    train(p.parse_args())
