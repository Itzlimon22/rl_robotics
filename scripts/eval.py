"""
eval.py — Held-out Transfer Evaluation Script
════════════════════════════════════════════════════════════════════════════════

Evaluates trained SAC policies and PID baseline on the held-out test
distribution (TEST_PARAM_CONFIG in auv_dr_wrapper.py). These are the
physics ranges NEVER seen during training — the main paper result.

Why separate from Cell 7?
    Cell 7 / EvalCallback evaluate on the TRAINING distribution.
    This script evaluates on the HELD-OUT TEST distribution.
    These are fundamentally different numbers and must not be confused.
    Only this script produces the transfer evaluation results for the paper.

Metrics reported:
    success_rate     — fraction of episodes where goal_dist < goal_threshold
    mean_reward      — mean total episode reward (± std)
    mean_dist        — mean final goal distance at episode end (± std)
    mean_energy      — mean absolute thrust per step (efficiency metric)
    total_energy     — mean total thrust effort per episode
    reward_std       — standard deviation of episode rewards (consistency)

Energy metric rationale:
    energy_per_step = mean(|actions|) averaged over all steps in episode
    Lower = more efficient. CDR trains with smooth reward → lower energy.
    PID uses brute-force gains → high energy even when successful.
    This metric is the key differentiator between CDR and PID in the paper.

Usage:
    # Evaluate SAC conditions
    python scripts/eval.py --mode curriculum --seed 1 --episodes 100
    python scripts/eval.py --mode uniform    --seed 1 --episodes 100
    python scripts/eval.py --mode none       --seed 1 --episodes 100

    # Evaluate all seeds for one condition
    python scripts/eval.py --mode curriculum --all-seeds --episodes 100

    # Evaluate all 9 trained models at once
    python scripts/eval.py --all --episodes 100

    # Evaluate PID baseline
    python scripts/eval.py --pid --episodes 100

    # Evaluate on harder custom test ranges
    python scripts/eval.py --mode curriculum --seed 1 --hard-test

    # Specify custom model directory
    python scripts/eval.py --mode curriculum --seed 1 --run-name curriculum_seed1

Output:
    Results printed to stdout and saved to JSON:
    ~/rl_research/auv/<mode>/<run_name>/test_eval_results.json

    Summary across seeds saved to:
    ~/rl_research/auv/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"

for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG, make_auv_env


# ─────────────────────────────────────────────────────────────────────────────
# Harder test distribution (optional, for stress testing)
# Push physics beyond TEST_PARAM_CONFIG to find breaking point.
# ─────────────────────────────────────────────────────────────────────────────
HARD_TEST_PARAM_CONFIG = {
    "c_drag_lateral": (0.60, 1.20),  # very high drag — twice the training max
    "c_drag_axial": (0.24, 0.48),
    "buoyancy_offset": (-0.15, 0.20),  # strong buoyancy disturbance
    "current_speed": (0.40, 0.80),  # strong current — 2-4x training max
    "added_mass": (0.30, 0.60),
    "act_efficiency": (0.60, 0.90),  # significantly degraded thrusters
}


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_base_dir() -> Path:
    """Find results base directory — Colab Drive or local."""
    on_colab = os.path.exists("/content/drive/MyDrive")
    if on_colab:
        return Path("/content/drive/MyDrive/rl_research/auv")
    return Path.home() / "rl_research" / "auv"


def resolve_xml_path() -> Path:
    """Find auv.xml in standard locations."""
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
    raise FileNotFoundError(
        "auv.xml not found. Place it in envs/auv.xml or pass --xml."
    )


def resolve_run_dir(base: Path, mode: str, run_name: str) -> Path:
    """Find the training run directory."""
    run_dir = base / mode / run_name
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Check that training completed and results are in {base}"
        )
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# SAC evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_sac(
    run_dir: Path,
    xml_path: Path,
    mode: str,
    n_episodes: int = 100,
    use_hard_test: bool = False,
    seed: int = 9999,
    verbose: bool = True,
) -> Dict:
    """
    Load a trained SAC model and evaluate on held-out test distribution.

    Parameters
    ----------
    run_dir : Path
        Directory containing best_model.zip and vec_normalize.pkl
    xml_path : Path
        Path to auv.xml
    mode : str
        DR mode used during training (none/uniform/curriculum)
    n_episodes : int
        Number of evaluation episodes
    use_hard_test : bool
        If True, use harder HARD_TEST_PARAM_CONFIG instead of TEST_PARAM_CONFIG
    seed : int
        RNG seed for evaluation environment
    verbose : bool
        Print progress during evaluation

    Returns
    -------
    dict with all metrics
    """
    # Import SB3 here to avoid import errors if not installed
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        raise ImportError(
            "stable-baselines3 required: pip install stable-baselines3[extra]"
        )

    model_path = run_dir / "best_model.zip"
    vecnorm_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    if verbose:
        print(f"\n[eval] Loading from: {run_dir}")

    # ── Build eval environment ────────────────────────────────────────────────
    def make_env():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        wrapper = AUVDomainRandomWrapper(env, mode=mode, seed=seed, verbose=False)
        # Switch to held-out test distribution
        wrapper.set_test_distribution()
        # If using harder ranges, override physics manually
        if use_hard_test:
            wrapper._test_mode = True
            # Monkey-patch _apply_test to use harder ranges
            import types

            def _apply_hard_test(self):
                return self._sample_and_apply(HARD_TEST_PARAM_CONFIG)

            wrapper._apply_test = types.MethodType(_apply_hard_test, wrapper)
        return wrapper

    vec_env = DummyVecEnv([make_env])

    # Load VecNormalize stats from training — CRITICAL for correct normalisation
    # The eval env must use the SAME running stats as the training env.
    # Otherwise the model sees observations in a different scale than it trained on.
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False  # freeze stats — do not update during eval
    vec_env.norm_reward = False  # raw rewards for interpretable metrics

    # Load model
    model = SAC.load(str(model_path), env=vec_env)

    if verbose:
        test_label = "HARD test" if use_hard_test else "TEST"
        print(
            f"[eval] Model loaded. Running {n_episodes} episodes on {test_label} distribution..."
        )
        if use_hard_test:
            print(
                f"[eval] Hard test ranges: drag_lateral={HARD_TEST_PARAM_CONFIG['c_drag_lateral']}, "
                f"current={HARD_TEST_PARAM_CONFIG['current_speed']}"
            )
        else:
            print(
                f"[eval] Test ranges: drag_lateral={TEST_PARAM_CONFIG['c_drag_lateral']}, "
                f"current={TEST_PARAM_CONFIG['current_speed']}"
            )

    # ── Run evaluation episodes ───────────────────────────────────────────────
    episode_rewards = []
    episode_dists = []
    episode_energies = []  # mean |action| per step — efficiency metric
    episode_total_energy = []  # sum |action| per episode — total effort
    episode_lengths = []
    successes = 0

    goal_threshold = 0.5  # must match auv_env.py goal_threshold

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_actions = []
        ep_steps = 0
        done = False
        final_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            ep_reward += float(reward[0])
            ep_actions.append(np.abs(action[0]))  # |action| for energy metric
            ep_steps += 1

            if done[0]:
                final_dist = info[0].get("goal_dist", float("inf"))

        # Record episode metrics
        episode_rewards.append(ep_reward)
        episode_dists.append(final_dist)
        episode_lengths.append(ep_steps)

        # Energy metrics
        if ep_actions:
            actions_array = np.array(ep_actions)  # shape (steps, 4)
            mean_energy = float(np.mean(actions_array))  # mean |thrust| per step
            total_energy = float(np.sum(actions_array))  # total effort
        else:
            mean_energy = 0.0
            total_energy = 0.0

        episode_energies.append(mean_energy)
        episode_total_energy.append(total_energy)

        # Success check
        if final_dist < goal_threshold:
            successes += 1

        # Progress reporting
        if verbose and (ep + 1) % 10 == 0:
            current_sr = successes / (ep + 1) * 100
            print(
                f"  {ep + 1}/{n_episodes} episodes | success so far: {current_sr:.0f}%"
            )

    vec_env.close()

    # ── Compute summary statistics ────────────────────────────────────────────
    results = {
        "mode": mode,
        "run_dir": str(run_dir),
        "n_episodes": n_episodes,
        "test_distribution": "hard" if use_hard_test else "standard",
        # Primary metrics
        "success_rate": float(successes / n_episodes),
        "success_count": int(successes),
        # Reward metrics
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        # Distance metrics
        "mean_dist": float(np.mean(episode_dists)),
        "std_dist": float(np.std(episode_dists)),
        # Energy metrics — KEY for paper comparison with PID
        "mean_energy_per_step": float(np.mean(episode_energies)),
        "std_energy_per_step": float(np.std(episode_energies)),
        "mean_total_energy": float(np.mean(episode_total_energy)),
        "std_total_energy": float(np.std(episode_total_energy)),
        # Episode length
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        # Raw data for custom analysis
        "episode_rewards": episode_rewards,
        "episode_dists": episode_dists,
        "episode_energies": episode_energies,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PID evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_pid(
    xml_path: Path,
    n_episodes: int = 100,
    use_hard_test: bool = False,
    seed: int = 9999,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate the PID baseline on held-out test distribution.

    Uses the same test physics ranges as SAC evaluation for fair comparison.
    PID gains are the tuned values from pid_baseline.py.

    Energy metric for PID will be much higher than SAC because PID uses
    brute-force proportional control that overshoots and oscillates.
    """
    if verbose:
        print(f"\n[eval] Running PID baseline for {n_episodes} episodes...")

    # PID gains — must match pid_baseline.py tuned values
    Kp = np.array([20.0, 20.0, 20.0])
    Ki = np.array([0.1, 0.1, 0.1])
    Kd = np.array([10.0, 10.0, 10.0])
    F_MAX = 20.0
    WINDUP_LIMIT = 8.0

    # PID uses larger goal threshold and more steps — fair for oscillatory control
    GOAL_THRESHOLD = 0.8  # larger than SAC's 0.5m
    MAX_STEPS = 1000  # 40 seconds — PID needs more time

    # Test ranges
    test_config = HARD_TEST_PARAM_CONFIG if use_hard_test else TEST_PARAM_CONFIG

    rng = np.random.default_rng(seed)

    episode_rewards = []
    episode_dists = []
    episode_energies = []
    episode_total_energy = []
    episode_lengths = []
    successes = 0

    for ep in range(n_episodes):
        # Create fresh environment each episode
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        wrapper = AUVDomainRandomWrapper(
            env, mode="uniform", seed=int(rng.integers(0, 10000)), verbose=False
        )

        # Manually sample test physics
        wrapper._sample_and_apply(test_config)

        obs, info = wrapper.reset()

        # PID state
        integral = np.zeros(3)
        prev_error = np.zeros(3)
        ep_reward = 0.0
        ep_actions = []
        ep_steps = 0

        for step in range(MAX_STEPS):
            # Get AUV and goal positions from info
            auv_pos = np.array(info.get("auv_pos", [0, 0, 0]))
            goal_pos = np.array(info.get("goal_pos", [5, 0, 0]))

            # Position error in world frame
            error = goal_pos - auv_pos
            dist = float(np.linalg.norm(error))

            # Check success
            if dist < GOAL_THRESHOLD:
                successes += 1
                break

            # PID computation
            derivative = error - prev_error
            integral += error
            # Anti-windup
            integral = np.clip(integral, -WINDUP_LIMIT, WINDUP_LIMIT)

            # World-frame force command
            F_world = Kp * error + Ki * integral + Kd * derivative

            # Simple thrust allocation: map world forces to 4 thrusters
            # Thrusters at rear in X-config — simplified allocation
            # Surge (X): all 4 thrusters equal
            # Heave (Z): top(+), bot(-) differential
            # Sway (Y):  left(+), right(-) differential
            Fx, Fy, Fz = F_world[0], F_world[1], F_world[2]

            thrust_top = np.clip((Fx + Fz) / (2 * F_MAX), -1.0, 1.0)
            thrust_bot = np.clip((Fx - Fz) / (2 * F_MAX), -1.0, 1.0)
            thrust_left = np.clip((Fx + Fy) / (2 * F_MAX), -1.0, 1.0)
            thrust_right = np.clip((Fx - Fy) / (2 * F_MAX), -1.0, 1.0)

            action = np.array(
                [thrust_top, thrust_bot, thrust_left, thrust_right], dtype=np.float32
            )

            prev_error = error.copy()

            # Step environment
            obs, reward, terminated, truncated, info = wrapper.step(action)
            ep_reward += float(reward)
            ep_actions.append(np.abs(action))
            ep_steps += 1

            if terminated or truncated:
                final_dist = info.get("goal_dist", dist)
                if final_dist < GOAL_THRESHOLD:
                    successes += 1
                break
        else:
            pass  # episode ended by MAX_STEPS

        final_dist = info.get("goal_dist", float("inf"))
        episode_rewards.append(ep_reward)
        episode_dists.append(final_dist)
        episode_lengths.append(ep_steps)

        if ep_actions:
            actions_array = np.array(ep_actions)
            episode_energies.append(float(np.mean(actions_array)))
            episode_total_energy.append(float(np.sum(actions_array)))
        else:
            episode_energies.append(0.0)
            episode_total_energy.append(0.0)

        wrapper.close()

        if verbose and (ep + 1) % 10 == 0:
            current_sr = successes / (ep + 1) * 100
            print(
                f"  {ep + 1}/{n_episodes} episodes | success so far: {current_sr:.0f}%"
            )

    results = {
        "mode": "pid",
        "n_episodes": n_episodes,
        "test_distribution": "hard" if use_hard_test else "standard",
        "success_rate": float(successes / n_episodes),
        "success_count": int(successes),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_dist": float(np.mean(episode_dists)),
        "std_dist": float(np.std(episode_dists)),
        "mean_energy_per_step": float(np.mean(episode_energies)),
        "std_energy_per_step": float(np.std(episode_energies)),
        "mean_total_energy": float(np.mean(episode_total_energy)),
        "std_total_energy": float(np.std(episode_total_energy)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "episode_rewards": episode_rewards,
        "episode_dists": episode_dists,
        "episode_energies": episode_energies,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results printing
# ─────────────────────────────────────────────────────────────────────────────


def print_results(results: Dict):
    """Print evaluation results in a clean readable format."""
    mode = results["mode"].upper()
    label = f"Mode: {mode}"
    if "seed" in results:
        label += f"  Seed: {results['seed']}"

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
        f"{results['std_energy_per_step']:.4f}  (lower = more efficient)"
    )
    print(
        f"  Total energy: {results['mean_total_energy']:.2f} ± "
        f"{results['std_total_energy']:.2f}  per episode"
    )
    print(
        f"  Episode len:  {results['mean_episode_length']:.1f} ± "
        f"{results['std_episode_length']:.1f} steps"
    )
    print(f"{'=' * 55}")


def save_results(results: Dict, save_path: Path):
    """Save results to JSON file."""
    # Remove raw episode data for smaller file (keep summary only)
    save_data = {
        k: v
        for k, v in results.items()
        if k not in ["episode_rewards", "episode_dists", "episode_energies"]
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[eval] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table across all conditions
# ─────────────────────────────────────────────────────────────────────────────


def print_summary_table(all_results: List[Dict]):
    """Print a clean comparison table of all conditions."""
    print(f"\n{'=' * 80}")
    print(f"  COMPLETE RESULTS SUMMARY — held-out test distribution")
    print(f"{'=' * 80}")
    print(
        f"  {'Condition':<20} {'Success':>8} {'Reward':>12} {'Dist':>8} {'Energy/step':>12}"
    )
    print(f"  {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 12}")

    for r in all_results:
        mode = r["mode"]
        seed = r.get("seed", "-")
        label = f"{mode} seed{seed}" if seed != "-" else mode
        print(
            f"  {label:<20} "
            f"{r['success_rate'] * 100:>7.1f}% "
            f"{r['mean_reward']:>8.2f}±{r['std_reward']:<6.2f} "
            f"{r['mean_dist']:>5.2f}m "
            f"{r['mean_energy_per_step']:>8.4f}±{r['std_energy_per_step']:.4f}"
        )

    print(f"{'=' * 80}")

    # Compute mean ± std across seeds for each mode
    modes = ["none", "uniform", "curriculum", "pid"]
    print(f"\n  MEAN ± STD ACROSS SEEDS:")
    print(f"  {'Condition':<15} {'Success':>10} {'Reward':>14} {'Energy/step':>14}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 14} {'-' * 14}")

    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue

        sr_values = [r["success_rate"] * 100 for r in mode_results]
        rew_values = [r["mean_reward"] for r in mode_results]
        eng_values = [r["mean_energy_per_step"] for r in mode_results]

        print(
            f"  {mode:<15} "
            f"{np.mean(sr_values):>6.1f}±{np.std(sr_values):<6.1f}% "
            f"{np.mean(rew_values):>7.2f}±{np.std(rew_values):<8.2f} "
            f"{np.mean(eng_values):>7.4f}±{np.std(eng_values):.4f}"
        )

    print(f"{'=' * 80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trained AUV policies on held-out test distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # What to evaluate
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["none", "uniform", "curriculum"],
        help="DR mode to evaluate. Required unless --pid or --all.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to evaluate. Required unless --all-seeds or --all.",
    )
    p.add_argument(
        "--all-seeds",
        action="store_true",
        help="Evaluate all seeds (0, 1, 2) for the given --mode.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 9 trained models (3 modes × 3 seeds).",
    )
    p.add_argument(
        "--pid", action="store_true", help="Evaluate PID baseline instead of SAC."
    )

    # Evaluation settings
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per model.",
    )
    p.add_argument(
        "--hard-test",
        action="store_true",
        help="Use harder test distribution (HARD_TEST_PARAM_CONFIG).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name. Default: <mode>_seed<seed>.",
    )

    # Paths
    p.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to auv.xml. Auto-detected if not specified.",
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base results directory. Auto-detected if not specified.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    xml_path = Path(args.xml) if args.xml else resolve_xml_path()
    base_dir = Path(args.base_dir) if args.base_dir else resolve_base_dir()

    print(f"[eval] XML:      {xml_path}")
    print(f"[eval] Base dir: {base_dir}")
    print(f"[eval] Episodes: {args.episodes}")
    print(f"[eval] Test dist: {'HARD' if args.hard_test else 'standard'}")

    all_results = []

    # ── PID evaluation ────────────────────────────────────────────────────────
    if args.pid:
        results = evaluate_pid(
            xml_path=xml_path,
            n_episodes=args.episodes,
            use_hard_test=args.hard_test,
            verbose=True,
        )
        print_results(results)

        save_path = base_dir / "pid_test_eval_results.json"
        save_results(results, save_path)
        return

    # ── Determine which models to evaluate ────────────────────────────────────
    if args.all:
        eval_list = [
            (m, s) for m in ["none", "uniform", "curriculum"] for s in [0, 1, 2]
        ]
    elif args.all_seeds:
        if args.mode is None:
            parser.error("--all-seeds requires --mode")
        eval_list = [(args.mode, s) for s in [0, 1, 2]]
    else:
        if args.mode is None or args.seed is None:
            parser.error(
                "Provide --mode and --seed, or use --all / --all-seeds / --pid"
            )
        eval_list = [(args.mode, args.seed)]

    # ── Run evaluations ───────────────────────────────────────────────────────
    for mode, seed in eval_list:
        run_name = args.run_name or f"{mode}_seed{seed}"

        # Handle v2 reruns for seed 0
        if seed == 0:
            v2_name = f"{mode}_seed0_v2"
            v2_dir = base_dir / mode / v2_name
            if v2_dir.exists():
                run_name = v2_name
                print(f"\n[eval] Using v2 rerun for seed 0: {run_name}")

        try:
            run_dir = resolve_run_dir(base_dir, mode, run_name)
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
                seed=9999 + seed,
                verbose=True,
            )
            results["seed"] = seed
            results["run_name"] = run_name

            print_results(results)

            # Save per-run results
            save_path = run_dir / "test_eval_results.json"
            save_results(results, save_path)

            all_results.append(results)

        except Exception as e:
            print(f"\n[eval] ERROR evaluating {mode} seed {seed}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # ── Print summary table if multiple models evaluated ──────────────────────
    if len(all_results) > 1:
        print_summary_table(all_results)

        # Save complete summary
        summary_path = base_dir / "eval_summary.json"
        summary_data = []
        for r in all_results:
            summary_data.append(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ["episode_rewards", "episode_dists", "episode_energies"]
                }
            )
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[eval] Complete summary saved → {summary_path}")


if __name__ == "__main__":
    main()
