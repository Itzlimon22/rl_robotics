"""
eval_perturbation.py — Phase 5: Sensor Glitch Recovery Evaluation
=================================================================
Loads a trained model and applies a temporary sensor failure
(heavy noise on rangefinders) mid-episode to evaluate robustness.
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
from envs.auv_dr_wrapper import make_auv_env, make_tracking_env, make_obstacle_env


def run_perturbation_eval(
    model_path: str,
    xml_path: str,
    mode: str,
    seed: int,
    task: str,
    n_episodes: int = 50,
):
    model_file = Path(model_path) / "final_model.zip"
    vec_file = Path(model_path) / "vec_normalize.pkl"

    if not model_file.exists() or not vec_file.exists():
        print(f"ERROR: Model or vec_normalize not found in {model_path}")
        sys.exit(1)

    print(f"\nLoading Model: {mode.upper()} (Task: {task}, Seed: {seed})")

    # 1. Setup Eval Environment
    def _init():
        if task == "obstacle":
            env = make_obstacle_env(xml_path, mode=mode, seed=seed)
        elif task == "tracking":
            env = make_tracking_env(xml_path, mode=mode, seed=seed)
        else:
            env = make_auv_env(xml_path, mode=mode, seed=seed)

        env.set_test_distribution()  # Test on hard physics
        return env

    venv = DummyVecEnv([_init])
    env = VecNormalize.load(str(vec_file), venv)
    env.training = False
    env.norm_reward = False

    model = SAC.load(str(model_file), env=env)

    # 2. Tracking Metrics
    successes = 0
    recovery_steps_list = []
    energy_used_list = []

    # 3. Evaluation Loop
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step_count = 0

        # Track recovery state
        pre_perturb_dist = 0.0
        is_recovering = False
        recovery_steps = 0
        ep_energy = 0.0

        while not done:
            # --- THE PERTURBATION TRIGGER (SENSOR GLITCH) ---
            # Between step 100 and 110, the rangefinders glitch and output noise
            if 100 <= step_count <= 110 and task == "obstacle":
                # Save actual pre-glitch distance on the first step of the glitch
                if step_count == 100:
                    base_env = env.venv.envs[0].env.unwrapped
                    pre_perturb_dist = base_env._goal_distance()
                    is_recovering = True

                # Corrupt the rangefinder data (last 4 indices) with heavy Gaussian noise
                obs_copy = obs.copy()
                obs_copy[0, -4:] += np.random.normal(0, 5.0, size=(4,))
                action, _ = model.predict(obs_copy, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            # --- TRACK RECOVERY ---
            base_env = env.venv.envs[0].env.unwrapped
            current_dist = base_env._goal_distance()

            if is_recovering:
                recovery_steps += 1
                # Recovered if we get back to the pre-glitch distance (after the glitch ends)
                if step_count > 110 and current_dist <= pre_perturb_dist:
                    is_recovering = False
                    recovery_steps_list.append(recovery_steps)

            # Step environment
            obs, reward, done, info = env.step(action)
            ep_energy += np.sum(np.abs(action))
            step_count += 1

            if done:
                if info[0].get("is_success", False):
                    successes += 1
                energy_used_list.append(ep_energy)
                # If it never recovered before episode ended
                if is_recovering:
                    recovery_steps_list.append(500)

    # 4. Print Results
    sr = (successes / n_episodes) * 100
    # Handle case where it never recovers
    if len(recovery_steps_list) == 0:
        mean_recovery = float("inf")
    else:
        mean_recovery = np.mean(recovery_steps_list)

    mean_energy = np.mean(energy_used_list)

    print("-" * 40)
    print(f"Results for {mode.upper()} under Sensor Glitch Perturbation")
    print("-" * 40)
    print(f"Success Rate:         {sr:.1f}%")
    print(f"Mean Recovery Steps:  {mean_recovery:.1f} steps")
    print(f"Mean Absolute Thrust: {mean_energy:.1f} N")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to folder containing final_model.zip",
    )
    parser.add_argument("--xml", type=str, default="envs/auv.xml")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["none", "uniform", "curriculum"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task", type=str, default="obstacle", choices=["base", "tracking", "obstacle"]
    )
    args = parser.parse_args()

    run_perturbation_eval(args.model_dir, args.xml, args.mode, args.seed, args.task)


if __name__ == "__main__":
    main()
