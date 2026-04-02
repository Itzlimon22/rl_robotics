"""
eval_perturbation.py — Mid-episode perturbation recovery evaluation
================================================================================
Applies a sudden impulse force at step T_perturb during each episode.
Supports both standard Goal-Reaching models and Trajectory Tracking models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

try:
    from auv_tracking_env import HalcyonAUVTrackingEnv
except ImportError:
    pass

# Constants for the Perturbation Test
N_EPISODES = 50
T_PERTURB = 100  # Apply impulse at exactly step 100
IMPULSE_N = 40.0  # Impulse force magnitude (N) — 2x max thruster force
IMPULSE_DUR = 5  # Impulse duration in steps (~0.2 seconds)


def apply_impulse(
    env_unwrapped: HalcyonAUVEnv, step_count: int, impulse_active: Dict
) -> bool:
    """Injects a force vector into the MuJoCo engine's applied forces tensor."""
    if T_PERTURB <= step_count < T_PERTURB + IMPULSE_DUR:
        if step_count == T_PERTURB:
            direction = np.random.standard_normal(3)
            direction[2] *= 0.2  # Dampen vertical throw
            direction /= np.linalg.norm(direction) + 1e-8
            impulse_active["dir"] = direction

        d = impulse_active.get("dir", np.array([1.0, 0.0, 0.0]))
        env_unwrapped.data.xfrc_applied[env_unwrapped._auv_body_id, 0:3] += (
            d * IMPULSE_N
        )
        return True
    return False


def evaluate_perturbation(args: argparse.Namespace):
    """Runs the perturbation test on a trained SAC policy."""
    on_colab = os.path.exists("/content/drive/MyDrive")
    base = (
        Path("/content/drive/MyDrive/rl_research/auv")
        if on_colab
        else Path.home() / "rl_research" / "auv"
    )

    # 1. Resolve correct folder based on task
    if args.is_tracking:
        run_name = args.run_name or f"tracking_{args.mode}_seed{args.seed}"
        mode_dir = f"tracking_{args.mode}"
    else:
        run_name = args.run_name or f"{args.mode}_seed{args.seed}"
        mode_dir = args.mode

    run_dir = base / mode_dir / run_name
    if not run_dir.exists():
        run_dir = base / mode_dir / f"{run_name}_v1"

    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find run directory: {run_dir}")

    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)

    # 2. Build the correct environment
    def _make_env():
        if args.is_tracking:
            env = HalcyonAUVTrackingEnv(xml_path=str(xml_path), path_speed=0.3)
        else:
            env = HalcyonAUVEnv(xml_path=str(xml_path))

        wrapper = AUVDomainRandomWrapper(env, mode="uniform", seed=args.seed + 9999)
        wrapper.set_test_distribution()  # Force hard test conditions
        return wrapper

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)
    base_env = vec_env.venv.envs[0].unwrapped

    label = f"TRACKING {args.mode}" if args.is_tracking else f"STATIC {args.mode}"
    print(
        f"\n[perturb] {label} seed={args.seed} | Impulse={IMPULSE_N}N at step {T_PERTURB}\n"
    )

    recovery_steps_list, post_success_list, energy_recovery_list = [], [], []

    obs = vec_env.reset()
    ep_count = step_count = recovery_steps = 0
    ep_energy = 0.0
    pre_perturb_dist = None
    perturbed = recovered = False
    impulse_active = {}

    while ep_count < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        apply_impulse(base_env, step_count, impulse_active)

        obs, reward, done, info = vec_env.step(action)
        ep_energy += float(np.mean(np.abs(action[0])))

        # 3. Track the correct metric (goal_dist vs tracking_error)
        dist_key = "tracking_error" if args.is_tracking else "goal_dist"
        current_dist = float(info[0].get(dist_key, float("inf")))

        if step_count == T_PERTURB - 1:
            pre_perturb_dist = current_dist
            perturbed = True
            recovered = False
            recovery_steps = 0

        if perturbed and not recovered and step_count >= T_PERTURB + IMPULSE_DUR:
            recovery_steps += 1
            # Recovery logic: Must get back within 1.0m OR 120% of pre-hit distance
            threshold = max(pre_perturb_dist * 1.2, 1.0)
            if pre_perturb_dist is not None and current_dist <= threshold:
                recovered = True

        if done[0]:
            success = (
                info[0].get("is_success", current_dist < 0.5)
                if args.is_tracking
                else current_dist < 0.5
            )
            post_success_list.append(float(success))
            recovery_steps_list.append(recovery_steps if recovered else 500)
            energy_recovery_list.append(ep_energy)

            ep_count += 1
            step_count = recovery_steps = 0
            ep_energy = 0.0
            pre_perturb_dist = None
            perturbed = recovered = False
            impulse_active = {}

            if ep_count % 10 == 0:
                print(
                    f"  {ep_count}/{N_EPISODES} | "
                    f"success={np.mean(post_success_list) * 100:.0f}% | "
                    f"mean_recovery={np.mean(recovery_steps_list):.1f} steps"
                )
        else:
            step_count += 1

    results = {
        "mode": args.mode,
        "seed": args.seed,
        "task": "tracking" if args.is_tracking else "static_goal",
        "impulse_N": IMPULSE_N,
        "post_perturbation_success_rate": float(np.mean(post_success_list)),
        "mean_recovery_steps": float(np.mean(recovery_steps_list)),
        "std_recovery_steps": float(np.std(recovery_steps_list)),
    }

    print(f"\n{'=' * 55}")
    print(f"  PERTURBATION RECOVERY — {label.upper()} seed={args.seed}")
    print(
        f"  Post-perturbation success: {results['post_perturbation_success_rate'] * 100:.1f}%"
    )
    print(
        f"  Mean recovery steps:       {results['mean_recovery_steps']:.1f} ± {results['std_recovery_steps']:.1f}"
    )
    print(f"{'=' * 55}\n")

    out = run_dir / "perturbation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate recovery from mid-episode physics perturbations."
    )
    p.add_argument("--mode", required=True, choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--is-tracking",
        action="store_true",
        help="Evaluate a tracking model instead of static goal",
    )
    p.add_argument("--run-name", type=str, default=None)
    return p


def main():
    evaluate_perturbation(build_parser().parse_args())


if __name__ == "__main__":
    main()
