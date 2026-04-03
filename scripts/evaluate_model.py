"""
evaluate_model.py — Quantitative Benchmarking
===================================================================
Loads a trained agent and runs N evaluation episodes to calculate
the true success rate and average path tracking error.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [_REPO_ROOT, _REPO_ROOT / "envs"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.auv_dr_wrapper import make_master_env


def evaluate(mode: str, seed: int, n_episodes: int = 100):
    run_dir = (
        Path.home()
        / "rl_research"
        / "auv"
        / f"master_{mode}"
        / f"master_{mode}_seed{seed}"
    )
    model_path = run_dir / "best_model.zip"
    vec_path = run_dir / "vec_normalize.pkl"
    xml_path = _REPO_ROOT / "envs" / "auv.xml"

    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return

    print(f"🔬 Loading Champion Model from Seed {seed}...")

    # Load Environment strictly in "none" mode for fair benchmarking,
    # but use a different seed to ensure it faces unseen initial conditions
    def _init():
        return make_master_env(str(xml_path), mode="none", seed=seed + 500)

    venv = DummyVecEnv([_init])
    env = VecNormalize.load(str(vec_path), venv)
    env.training = False
    env.norm_reward = False

    model = SAC.load(str(model_path), env=env)

    print(f"⏱️ Running {n_episodes} evaluation episodes. Please wait...\n")

    successes = 0
    total_rewards = []
    tracking_errors = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        final_dist = -1.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]

            # Extract tracking distance from the final step's info dictionary
            if done:
                final_dist = info[0].get("goal_dist", -1.0)

        total_rewards.append(ep_reward)
        tracking_errors.append(final_dist)

        # Consider it a success if it finished the episode within 0.5m of the path
        if 0 < final_dist < 0.5:
            successes += 1

    # Print Final Analytics
    success_rate = (successes / n_episodes) * 100
    mean_reward = np.mean(total_rewards)
    mean_error = np.mean([e for e in tracking_errors if e > 0])

    print("-" * 40)
    print(f"📊 FINAL ANALYTICS (N={n_episodes} Episodes)")
    print("-" * 40)
    print(f"🏆 Success Rate:        {success_rate:.1f}%")
    print(f"💰 Mean Reward:         {mean_reward:.2f}")
    print(f"📏 Avg Tracking Error:  {mean_error:.3f} meters")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="curriculum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.mode, args.seed, args.episodes)
