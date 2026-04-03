"""
generate_table.py — Automated Results Exporter
===================================================================
Evaluates all trained AUV models (None, Uniform, Curriculum) across
all seeds and exports the final metrics to a publication-ready CSV.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [_REPO_ROOT, _REPO_ROOT / "envs"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.auv_dr_wrapper import make_master_env


def evaluate_model(mode: str, seed: int, n_episodes: int = 100):
    """
    Evaluates a specific model and returns Success Rate, Mean Reward, and Error.
    """
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

    # Early Return if model doesn't exist yet
    if not model_path.exists():
        print(f"⚠️ Warning: Model {mode} (Seed {seed}) not found. Skipping.")
        return None, None, None

    def _init():
        return make_master_env(str(xml_path), mode="none", seed=seed + 500)

    venv = DummyVecEnv([_init])
    env = VecNormalize.load(str(vec_path), venv)
    env.training = False
    env.norm_reward = False

    model = SAC.load(str(model_path), env=env)

    successes = 0
    total_rewards = []
    tracking_errors = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        final_dist = -1.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]
            if done:
                final_dist = info[0].get("goal_dist", -1.0)

        total_rewards.append(ep_reward)
        tracking_errors.append(final_dist)
        if 0 < final_dist < 0.5:
            successes += 1

    success_rate = (successes / n_episodes) * 100
    mean_reward = np.mean(total_rewards)
    mean_error = np.mean([e for e in tracking_errors if e > 0])

    return success_rate, mean_reward, mean_error


def main():
    modes = ["none", "uniform", "curriculum"]
    seeds = [0, 1, 2, 3]  # Assuming 4 seeds total
    results = []

    print("🚀 Starting Automated Benchmarking...")

    for mode in modes:
        for seed in seeds:
            print(f"  Evaluating {mode.upper()} - Seed {seed}...")
            sr, mr, me = evaluate_model(mode, seed, n_episodes=100)

            if sr is not None:
                results.append(
                    {
                        "Algorithm": f"SAC-{mode.capitalize()}",
                        "Seed": seed,
                        "Success Rate (%)": round(sr, 1),
                        "Mean Reward": round(mr, 2),
                        "Cross-Track Error (m)": round(me, 3),
                    }
                )

    # Save to CSV
    df = pd.DataFrame(results)
    out_dir = Path.home() / "rl_research" / "paper_assets" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Evaluation_Results.csv"

    df.to_csv(out_file, index=False)
    print(f"✅ Full benchmark table saved to: {out_file}")


if __name__ == "__main__":
    main()
