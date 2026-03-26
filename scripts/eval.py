"""
eval.py — Load a saved model from Drive or local and evaluate it.

Usage:
  python scripts/eval.py --env HalfCheetah-v4 --run-name run_01
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def get_save_dir(env_id, run_name):
    gdrive = "/content/drive/MyDrive/rl_research"
    local  = os.path.expanduser("~/rl_research")
    base   = gdrive if os.path.exists("/content/drive") else local
    return os.path.join(base, env_id, run_name)


def evaluate(args):
    save_dir = get_save_dir(args.env, args.run_name)
    AlgoCls  = SAC if args.algo == "sac" else TD3

    # Load env + normalizer
    env = make_vec_env(args.env, n_envs=1)
    norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training    = False
        env.norm_reward = False
        print("✅  Loaded observation normalizer")

    # Load best model
    model_path = os.path.join(save_dir, "best_model", "best_model")
    model = AlgoCls.load(model_path, env=env)
    print(f"✅  Loaded model from {model_path}\n")

    # Evaluate
    rewards = []
    for ep in range(args.n_episodes):
        obs = env.reset()
        ep_reward, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
        rewards.append(ep_reward)
        print(f"  Episode {ep+1:2d}: {ep_reward:8.1f}")

    print(f"\nMean ± Std: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Min / Max:  {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env",        default="HalfCheetah-v4")
    p.add_argument("--algo",       default="sac", choices=["sac", "td3"])
    p.add_argument("--run-name",   default="run_01")
    p.add_argument("--n-episodes", type=int, default=10)
    evaluate(p.parse_args())
