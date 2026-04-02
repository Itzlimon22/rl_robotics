"""
eval_tracking.py — Evaluation script for Phase 3 Trajectory Tracking
=====================================================================
Loads a trained SAC model from the tracking task and evaluates it.
Reports: success rate, tracking error, energy per step.

Usage:
    python scripts/eval_tracking.py --mode curriculum --seed 0
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
from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

N_EVAL_EPISODES = 50


def find_run_dir(mode: str, seed: int, base: Path) -> Path:
    """Resolves the directory for the tracking models."""
    run_name = f"tracking_{mode}"
    run_dir = base / run_name / f"{run_name}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def find_xml() -> Path:
    candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found")


def make_eval_env(xml_path: str, seed: int) -> VecNormalize:
    """Instantiates the tracking environment with test distribution."""

    def _init():
        env = HalcyonAUVTrackingEnv(xml_path=str(xml_path), path_speed=0.3)
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=seed + 9999)
        env.set_test_distribution()
        return env

    return DummyVecEnv([_init])


def run_eval_loop(model: SAC, vec_env: VecNormalize, n_episodes: int) -> dict:
    """Executes the evaluation loop and aggregates metrics."""
    successes, tracking_errors = [], []
    energies, ep_lengths = [], []

    obs = vec_env.reset()
    ep_energy = 0.0
    ep_steps = 0
    ep_count = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Track energy effort
        raw_action = action[0]
        action_mag = float(np.mean(np.abs(raw_action)))
        ep_energy += action_mag
        ep_steps += 1

        if done[0]:
            # Success in tracking means we didn't terminate early due to max error
            tracking_err = float(info[0].get("mean_tracking_error", 5.0))
            success = tracking_err < 1.0  # Kept average error under 1m

            successes.append(float(success))
            tracking_errors.append(tracking_err)
            energies.append(ep_energy / max(ep_steps, 1))
            ep_lengths.append(ep_steps)

            ep_count += 1
            ep_energy = 0.0
            ep_steps = 0

            if ep_count % 10 == 0:
                print(
                    f"  {ep_count}/{n_episodes} | "
                    f"success={np.mean(successes) * 100:.0f}% | "
                    f"err={np.mean(tracking_errors):.2f}m"
                )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_tracking_error": float(np.mean(tracking_errors)),
        "mean_energy_per_step": float(np.mean(energies)),
        "mean_episode_length": float(np.mean(ep_lengths)),
        "n_episodes": n_episodes,
    }


def evaluate_tracking(args: argparse.Namespace):
    on_colab = Path("/content/drive/MyDrive").exists()
    base = (
        Path("/content/drive/MyDrive/rl_research/auv")
        if on_colab
        else Path.home() / "rl_research" / "auv"
    )

    run_dir = find_run_dir(args.mode, args.seed, base)
    xml_path = find_xml()

    print(f"[eval_tracking] Loading: {run_dir}")

    vec_env = make_eval_env(xml_path, args.seed)
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)

    print(f"\n[eval_tracking] Running {N_EVAL_EPISODES} episodes...\n")
    results = run_eval_loop(model, vec_env, N_EVAL_EPISODES)

    print(f"\n{'=' * 55}")
    print(f"  TRACKING EVAL — {args.mode.upper()} seed={args.seed}")
    print(f"  Success rate:    {results['success_rate'] * 100:.1f}%")
    print(f"  Tracking Error:  {results['mean_tracking_error']:.2f}m")
    print(f"  Energy/step:     {results['mean_energy_per_step']:.4f}")
    print(f"{'=' * 55}\n")

    out = run_dir / "test_tracking_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_tracking] Saved → {out}")
    vec_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "uniform", "curriculum"], required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate_tracking(args)


if __name__ == "__main__":
    main()
