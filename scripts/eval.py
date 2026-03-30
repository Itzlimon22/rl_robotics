"""
eval.py — Held-out transfer evaluation for Halcyon AUV
=======================================================
Evaluates trained models on TEST_PARAM_CONFIG (held-out physics ranges).
This is the main paper result.

Usage:
    python scripts/eval.py --mode curriculum --seed 0
    python scripts/eval.py --mode uniform --seed 1
    python scripts/eval.py --mode none --seed 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper


N_EVAL_EPISODES = 50


def find_run_dir(mode: str, seed: int, base: Path) -> Path:
    run_dir = base / mode / f"{mode}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def evaluate(args: argparse.Namespace):
    on_colab = Path("/content/drive/MyDrive").exists()
    if on_colab:
        base = Path("/content/drive/MyDrive/rl_research/auv")
    else:
        base = Path.home() / "rl_research" / "auv"

    run_dir = find_run_dir(args.mode, args.seed, base)
    print(f"[eval] Loading from: {run_dir}")

    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        raise FileNotFoundError("auv.xml not found")

    def _make_env():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=args.seed + 9999)
        env.set_test_distribution()
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)
    print(
        f"[eval] Model loaded. Running {N_EVAL_EPISODES} episodes on TEST distribution..."
    )
    print(f"[eval] Test ranges: drag_lateral=[0.30,0.80], current=[0.20,0.60]")

    successes = []
    rewards = []
    final_dists = []

    obs = vec_env.reset()
    ep_reward = 0.0
    ep_count = 0

    while ep_count < N_EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        ep_reward += reward[0]

        if done[0]:
            goal_dist = info[0].get("goal_dist", float("inf"))
            success = goal_dist < 0.5
            successes.append(float(success))
            rewards.append(ep_reward)
            final_dists.append(goal_dist)
            ep_reward = 0.0
            ep_count += 1
            if ep_count % 10 == 0:
                print(
                    f"  {ep_count}/{N_EVAL_EPISODES} episodes | "
                    f"success so far: {np.mean(successes) * 100:.0f}%"
                )

    success_rate = np.mean(successes)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_dist = np.mean(final_dists)
    std_dist = np.std(final_dists)

    print(f"\n{'=' * 55}")
    print(f"  TRANSFER EVAL — held-out test distribution")
    print(f"  Mode: {args.mode}  Seed: {args.seed}")
    print(
        f"  Success rate: {success_rate * 100:.1f}%  "
        f"({int(sum(successes))}/{N_EVAL_EPISODES})"
    )
    print(f"  Mean reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean dist:    {mean_dist:.2f}m  +/- {std_dist:.2f}m")
    print(f"{'=' * 55}\n")

    results = {
        "mode": args.mode,
        "seed": args.seed,
        "eval_type": "held_out_test",
        "n_episodes": N_EVAL_EPISODES,
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_dist": float(mean_dist),
        "std_dist": float(std_dist),
    }
    out_path = run_dir / "test_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved → {out_path}")

    vec_env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
