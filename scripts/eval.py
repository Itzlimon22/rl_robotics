"""
eval.py — Held-out Transfer Evaluation Script
================================================================================
Evaluates trained SAC models and PID baselines on the AUV environment.
Calculates success rates, reward statistics, and critical hardware metrics
(Energy per Step and Peak Thrust) necessary for sim-to-real transfer validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"

for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG

HARD_TEST_PARAM_CONFIG = {
    "c_drag_lateral": (0.60, 1.20),
    "c_drag_axial": (0.24, 0.48),
    "buoyancy_offset": (-0.15, 0.20),
    "current_speed": (0.40, 0.80),
    "added_mass": (0.30, 0.60),
    "act_efficiency": (0.60, 0.90),
}


def resolve_base_dir() -> Path:
    """Resolves the base directory depending on local vs Colab execution."""
    on_colab = os.path.exists("/content/drive/MyDrive")
    if on_colab:
        return Path("/content/drive/MyDrive/rl_research/auv")
    return Path.home() / "rl_research" / "auv"


def resolve_xml_path() -> Path:
    """Locates the auv.xml file across common directories."""
    candidates = [
        _ENVS_DIR / "auv.xml",
        _REPO_ROOT / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("/content/rl_robotics/envs/auv.xml"),
        Path("auv.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found.")


def resolve_run_dir(base: Path, target_mode: str, run_name: str) -> Path:
    """
    Finds the training run directory, prioritizing '_v1' resumed runs
    which contain more training steps and better performance.
    """
    has_suffix = any(run_name.endswith(s) for s in ["_v1", "_v2", "_v3"])
    if has_suffix:
        run_dir = base / target_mode / run_name
        if run_dir.exists():
            return run_dir
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    for suffix in ["_v1", ""]:
        candidate = base / target_mode / f"{run_name}{suffix}"
        if candidate.exists() and (candidate / "best_model.zip").exists():
            if suffix == "_v1":
                print(
                    f"[eval] Found resumed run: {candidate.name} (preferred over original)"
                )
            return candidate

    mode_dir = base / target_mode
    available = []
    if mode_dir.exists():
        available = sorted(
            [
                d.name
                for d in mode_dir.iterdir()
                if d.is_dir() and (d / "best_model.zip").exists()
            ]
        )
    raise FileNotFoundError(
        f"No run found for {run_name}\nAvailable in {mode_dir}: {available}"
    )


def evaluate_sac(
    run_dir: Path,
    xml_path: Path,
    mode: str,
    n_episodes: int = 100,
    use_hard_test: bool = False,
    use_obstacle: bool = False,
    use_extreme_test: bool = False,
    seed: int = 9999,
    verbose: bool = True,
) -> Dict:
    """
    Evaluates a trained SAC model, tracking success, reward, energy, and peak thrust.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        raise ImportError("pip install stable-baselines3[extra]")

    model_path = run_dir / "best_model.zip"
    vecnorm_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    if verbose:
        print(f"\n[eval] Loading from: {run_dir}")

    test_config = HARD_TEST_PARAM_CONFIG if use_hard_test else TEST_PARAM_CONFIG

    def make_env():
        env = HalcyonAUVEnv(xml_path=str(xml_path))

        # Inject the Obstacle Wrapper if flag is passed
        if use_obstacle:
            from auv_obstacle_env import ObstacleAUVWrapper

            env = ObstacleAUVWrapper(env)

        wrapper = AUVDomainRandomWrapper(env, mode=mode, seed=seed, verbose=False)

        # Apply extreme test conditions if requested
        if use_extreme_test:
            wrapper.set_extreme_test_distribution()
        else:
            wrapper._sample_and_apply(test_config)
        return wrapper

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(model_path), env=vec_env)

    if verbose:
        label = "HARD test" if use_hard_test else "TEST"
        if use_extreme_test:
            label = "EXTREME (200% bounds)"
        print(
            f"[eval] Model loaded. Running {n_episodes} episodes on {label} distribution..."
        )
        cfg = test_config
        if not use_extreme_test:
            print(
                f"[eval] Test ranges: drag_lateral={cfg['c_drag_lateral']}, "
                f"current={cfg['current_speed']}"
            )

    episode_rewards, episode_dists = [], []
    episode_energies, episode_lengths, peak_thrusts = [], [], []
    successes = 0

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_energy = 0.0
        ep_peak = 0.0
        ep_steps = 0
        done = False
        final_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            raw_action = action[0]
            action_mag = float(np.mean(np.abs(raw_action)))

            ep_energy += action_mag
            ep_peak = max(ep_peak, float(np.max(np.abs(raw_action))))
            ep_reward += float(reward[0])
            ep_steps += 1

            if done[0]:
                final_dist = info[0].get("goal_dist", float("inf"))

        episode_rewards.append(ep_reward)
        episode_dists.append(final_dist)
        episode_lengths.append(ep_steps)
        episode_energies.append(ep_energy / max(ep_steps, 1))
        peak_thrusts.append(ep_peak)

        if final_dist < 0.5:
            successes += 1

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"  {ep + 1}/{n_episodes} episodes | success so far: {successes / (ep + 1) * 100:.0f}%"
            )

    vec_env.close()

    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "n_episodes": n_episodes,
        "test_distribution": "hard" if use_hard_test else "standard",
        "success_rate": float(successes / n_episodes),
        "success_count": int(successes),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_dist": float(np.mean(episode_dists)),
        "std_dist": float(np.std(episode_dists)),
        "mean_energy_per_step": float(np.mean(episode_energies)),
        "std_energy_per_step": float(np.std(episode_energies)),
        "mean_peak_thrust": float(np.mean(peak_thrusts)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def evaluate_pid(
    xml_path: Path,
    n_episodes: int = 100,
    use_hard_test: bool = False,
    use_obstacle: bool = False,
    use_extreme_test: bool = False,
    seed: int = 9999,
    kp: float = 20.0,
    verbose: bool = True,
) -> Dict:
    """
    Evaluates the standard PID controller baseline, matching SAC metrics.
    """
    if verbose:
        print(f"\n[eval] Running PID baseline for {n_episodes} episodes...")

    Kp = np.array([kp, kp, kp])  # Use custom kp value
    Ki = np.array([0.1, 0.1, 0.1])
    Kd = np.array([10.0, 10.0, 10.0])
    F_MAX = 20.0
    WINDUP_LIMIT = 8.0
    GOAL_THRESHOLD = 0.8
    MAX_STEPS = 1000

    test_config = HARD_TEST_PARAM_CONFIG if use_hard_test else TEST_PARAM_CONFIG
    if use_extreme_test:
        from auv_dr_wrapper import EXTREME_TEST_CONFIG

        test_config = EXTREME_TEST_CONFIG
    rng = np.random.default_rng(seed)

    episode_rewards, episode_dists = [], []
    episode_energies, episode_lengths, peak_thrusts = [], [], []
    successes = 0

    for ep in range(n_episodes):
        env = HalcyonAUVEnv(xml_path=str(xml_path))

        if use_obstacle:
            from auv_obstacle_env import ObstacleAUVWrapper

            env = ObstacleAUVWrapper(env)

        wrapper = AUVDomainRandomWrapper(
            env, mode="uniform", seed=int(rng.integers(0, 10000)), verbose=False
        )
        wrapper._sample_and_apply(test_config)
        obs, info = wrapper.reset()

        integral = np.zeros(3)
        prev_error = np.zeros(3)
        ep_reward = 0.0
        ep_energy = 0.0
        ep_peak = 0.0
        ep_steps = 0
        ep_success = False

        for step in range(MAX_STEPS):
            auv_pos = np.array(info.get("auv_pos", [0, 0, 0]))
            goal_pos = np.array(info.get("goal_pos", [5, 0, 0]))
            error = goal_pos - auv_pos
            dist = float(np.linalg.norm(error))

            if dist < GOAL_THRESHOLD:
                ep_success = True
                break

            derivative = error - prev_error
            integral = np.clip(integral + error, -WINDUP_LIMIT, WINDUP_LIMIT)
            F_world = Kp * error + Ki * integral + Kd * derivative

            Fx, Fy, Fz = F_world
            action = np.clip(
                np.array(
                    [
                        (Fx + Fz) / (2 * F_MAX),
                        (Fx - Fz) / (2 * F_MAX),
                        (Fx + Fy) / (2 * F_MAX),
                        (Fx - Fy) / (2 * F_MAX),
                    ],
                    dtype=np.float32,
                ),
                -1.0,
                1.0,
            )

            action_mag = float(np.mean(np.abs(action)))
            ep_energy += action_mag
            ep_peak = max(ep_peak, float(np.max(np.abs(action))))

            prev_error = error.copy()
            obs, reward, terminated, truncated, info = wrapper.step(action)
            ep_reward += float(reward)
            ep_steps += 1

            if terminated or truncated:
                if info.get("goal_dist", dist) < GOAL_THRESHOLD:
                    ep_success = True
                break

        if ep_success:
            successes += 1

        final_dist = info.get("goal_dist", float("inf"))
        episode_rewards.append(ep_reward)
        episode_dists.append(final_dist)
        episode_lengths.append(ep_steps)
        episode_energies.append(ep_energy / max(ep_steps, 1))
        peak_thrusts.append(ep_peak)

        wrapper.close()

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"  {ep + 1}/{n_episodes} | success: {successes / (ep + 1) * 100:.0f}%"
            )

    return {
        "mode": "pid",
        "n_episodes": n_episodes,
        "test_distribution": "hard" if use_hard_test else "standard",
        "success_rate": float(successes / n_episodes),
        "success_count": int(successes),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_dist": float(np.mean(episode_dists)),
        "std_dist": float(np.std(episode_dists)),
        "mean_energy_per_step": float(np.mean(episode_energies)),
        "std_energy_per_step": float(np.std(episode_energies)),
        "mean_peak_thrust": float(np.mean(peak_thrusts)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def print_results(results: Dict):
    """Prints a cleanly formatted breakdown of an individual evaluation run."""
    mode = results["mode"].upper()
    seed = results.get("seed", "-")
    label = f"{mode} seed={seed}" if seed != "-" else mode
    print(f"\n{'=' * 55}")
    print(f"  TRANSFER EVAL — held-out test distribution")
    print(f"  {label}")
    print(
        f"  Success rate: {results['success_rate'] * 100:.1f}%  "
        f"({results['success_count']}/{results['n_episodes']})"
    )
    print(f"  Mean reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean dist:    {results['mean_dist']:.2f}m ± {results['std_dist']:.2f}m")
    print(
        f"  Energy/step:  {results['mean_energy_per_step']:.4f} ± "
        f"{results['std_energy_per_step']:.4f}"
    )
    print(f"  Peak thrust:  {results['mean_peak_thrust']:.4f}")
    print(f"  Episode len:  {results['mean_episode_length']:.1f} steps")
    print(f"{'=' * 55}")


def save_results(results: Dict, save_path: Path):
    """Saves evaluation metrics to JSON for visualization scripts."""
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Saved → {save_path}")


def print_summary_table(all_results: List[Dict]):
    """Prints an aggregated table of all models tested in the current batch."""
    print(f"\n{'=' * 85}")
    print(f"  COMPLETE RESULTS SUMMARY")
    print(
        f"  {'Condition':<22} {'Success':>8} {'Reward':>12} {'Dist':>8} {'Energy/step':>12} {'Peak Thrust':>12}"
    )
    print(f"  {'-' * 22} {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 12}")
    for r in all_results:
        label = f"{r.get('target_mode', r['mode'])} s{r.get('seed', '-')}"
        print(
            f"  {label:<22} "
            f"{r['success_rate'] * 100:>7.1f}% "
            f"{r['mean_reward']:>7.2f}±{r['std_reward']:<6.2f} "
            f"{r['mean_dist']:>5.2f}m "
            f"{r['mean_energy_per_step']:>8.4f} "
            f"{r['mean_peak_thrust']:>11.4f}"
        )

    print(f"\n  MEAN ± STD ACROSS SEEDS:")
    for mode in ["none", "uniform", "curriculum", "pid"]:
        mrs = [r for r in all_results if r["mode"] == mode]
        if not mrs:
            continue
        sr = [r["success_rate"] * 100 for r in mrs]
        rw = [r["mean_reward"] for r in mrs]
        en = [r["mean_energy_per_step"] for r in mrs]
        pt = [r["mean_peak_thrust"] for r in mrs]
        print(
            f"  {mode:<15} "
            f"{np.mean(sr):>6.1f}±{np.std(sr):<5.1f}% "
            f"{np.mean(rw):>7.2f}±{np.std(rw):<7.2f} "
            f"{np.mean(en):>7.4f}±{np.std(en):.4f} "
            f"{np.mean(pt):>9.4f}±{np.std(pt):.4f}"
        )
    print(f"{'=' * 85}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trained AUV policies on held-out test distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", type=str, default=None, choices=["none", "uniform", "curriculum"]
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--all", action="store_true", help="Evaluate all 9 trained models")
    p.add_argument("--pid", action="store_true")
    p.add_argument(
        "--master", action="store_true", help="Evaluate master runs with sensor noise"
    )
    p.add_argument(
        "--obstacle",
        action="store_true",
        help="Enable ObstacleAUVWrapper (23-dim observation space)",
    )
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--hard-test", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--xml", type=str, default=None)
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument(
        "--kp", type=float, default=20.0, help="Proportional gain for PID controller"
    )
    p.add_argument(
        "--extreme-test",
        action="store_true",
        help="Use 2x out-of-distribution parameters to test sim-to-real proxy bounds.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    xml_path = Path(args.xml) if args.xml else resolve_xml_path()
    base_dir = Path(args.base_dir) if args.base_dir else resolve_base_dir()

    print(f"[eval] XML:      {xml_path}")
    print(f"[eval] Base dir: {base_dir}")
    print(f"[eval] Episodes: {args.episodes}")
    if args.extreme_test:
        print(f"[eval] Test dist: EXTREME (200% bounds)")
    else:
        print(f"[eval] Test dist: {'HARD' if args.hard_test else 'standard'}")
    if args.master:
        print(f"[eval] Master run execution enabled.")
    if args.obstacle:
        print(f"[eval] Obstacle environment enabled (23-dim).")

    all_results = []

    if args.pid:
        results = evaluate_pid(
            xml_path=xml_path,
            n_episodes=args.episodes,
            use_hard_test=args.hard_test,
            use_obstacle=args.obstacle,
            use_extreme_test=args.extreme_test,
            kp=args.kp,
            verbose=True,
        )
        print_results(results)
        save_results(results, base_dir / "pid_test_eval_results.json")
        return

    if args.all:
        eval_list = [
            (m, s) for m in ["none", "uniform", "curriculum"] for s in [0, 1, 2]
        ]
    elif args.all_seeds:
        if not args.mode:
            parser.error("--all-seeds requires --mode")
        eval_list = [(args.mode, s) for s in [0, 1, 2]]
    else:
        if not args.mode or args.seed is None:
            parser.error("Provide --mode and --seed, or --all / --all-seeds / --pid")
        eval_list = [(args.mode, args.seed)]

    for mode, seed in eval_list:
        target_mode = f"master_{mode}" if args.master else mode
        run_name = args.run_name or f"{target_mode}_seed{seed}"

        try:
            run_dir = resolve_run_dir(base_dir, target_mode, run_name)
        except FileNotFoundError as e:
            print(f"\n[eval] SKIP — {e}")
            continue

        try:
            results = evaluate_sac(
                run_dir=run_dir,
                xml_path=xml_path,
                mode=mode,
                n_episodes=args.episodes,
                use_hard_test=args.hard_test,
                use_obstacle=args.obstacle,
                use_extreme_test=args.extreme_test,
                seed=9999 + seed,
                verbose=True,
            )
            results["seed"] = seed
            results["run_name"] = run_dir.name
            results["target_mode"] = target_mode
            print_results(results)
            save_results(results, run_dir / "test_eval_results.json")
            all_results.append(results)
        except Exception as e:
            print(f"\n[eval] ERROR {target_mode} seed{seed}: {e}")
            import traceback

            traceback.print_exc()

    if len(all_results) > 1:
        print_summary_table(all_results)
        out = base_dir / "eval_summary.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[eval] Summary saved → {out}")


if __name__ == "__main__":
    main()
