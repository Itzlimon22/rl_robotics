"""
eval_tracking.py — Correct evaluation for trajectory tracking models
=====================================================================
Tracking success criterion: mean_tracking_error < 1.0m (not goal_dist < 0.5m)
The agent must follow a lemniscate path, not reach a static goal.

Why original eval showed 0%:
  eval.py checks goal_dist < 0.5m — tracking env has no static goal,
  so this ALWAYS returns 0% regardless of actual tracking performance.

This script uses the correct criterion: mean distance from path < threshold.

Usage:
    python scripts/eval_tracking.py --mode tracking_curriculum --seed 0
    python scripts/eval_tracking.py --all    # all 3 tracking runs
"""

from __future__ import annotations
import argparse, json, sys
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

N_EPISODES = 50
TRACKING_SUCCESS_THRESHOLD = 1.0  # metres — mean path error for "success"
GOAL_SUCCESS_THRESHOLD = 0.5  # metres — for goal-reaching fallback


def resolve_base():
    from pathlib import Path
    import os

    # Check local data directory first
    local_data = Path.home() / "rl_robotics" / "data"
    if (local_data / "tracking").exists():
        return local_data / "tracking"

    colab = Path("/content/drive/MyDrive/rl_research/auv")
    local = Path.home() / "rl_research" / "auv"
    return colab if colab.exists() else local


def find_xml():
    candidates = [
        _ENVS_DIR / "auv.xml",
        _REPO_ROOT / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("/content/rl_robotics/envs/auv.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found")


def find_run_dir(base, mode, seed, run_name=None):
    """Find run directory. If run_name provided, use it directly. Otherwise search for mode_seedN."""
    if run_name:
        # Direct custom run name: look for base/mode/run_name
        rd = base / mode / run_name
        if rd.exists() and (rd / "best_model.zip").exists():
            return rd
        raise FileNotFoundError(
            f"Custom run not found: {rd}. Check --run-name and --base-dir."
        )

    # Original search logic: look for mode_seedN with optional suffixes
    for suffix in ["_v1", ""]:
        rd = base / mode / f"{mode}_seed{seed}{suffix}"
        if rd.exists() and (rd / "best_model.zip").exists():
            return rd
    raise FileNotFoundError(f"No run found: {mode} seed{seed}")


def make_tracking_env(xml_path, seed, use_extreme_test=False):
    """
    Build tracking environment.
    Falls back to goal-reaching if auv_tracking_env.py not available.
    """
    try:
        from auv_tracking_env import HalcyonAUVTrackingEnv
        from auv_dr_wrapper import (
            AUVDomainRandomWrapper,
            TEST_PARAM_CONFIG,
            EXTREME_TEST_CONFIG,
        )

        def _init():
            env = HalcyonAUVTrackingEnv(
                xml_path=str(xml_path),
                path_speed=0.3,
                tracking_threshold=TRACKING_SUCCESS_THRESHOLD,
            )
            wrapper = AUVDomainRandomWrapper(
                env, mode="uniform", seed=seed + 9999, verbose=False
            )
            if use_extreme_test:
                wrapper.set_extreme_test_distribution()
            else:
                wrapper._sample_and_apply(TEST_PARAM_CONFIG)
            return wrapper

        return DummyVecEnv([_init]), "tracking"

    except ImportError:
        # Fallback: use regular goal-reaching env
        print("[eval_tracking] WARNING: auv_tracking_env.py not found")
        print("[eval_tracking] Using goal-reaching env as fallback")
        from auv_env import HalcyonAUVEnv
        from auv_dr_wrapper import (
            AUVDomainRandomWrapper,
            TEST_PARAM_CONFIG,
            EXTREME_TEST_CONFIG,
        )

        def _init():
            env = HalcyonAUVEnv(xml_path=str(xml_path))
            wrapper = AUVDomainRandomWrapper(
                env, mode="uniform", seed=seed + 9999, verbose=False
            )
            if use_extreme_test:
                wrapper.set_extreme_test_distribution()
            else:
                wrapper._sample_and_apply(TEST_PARAM_CONFIG)
            return wrapper

        return DummyVecEnv([_init]), "goal_reaching"


def evaluate_tracking(
    mode, seed, n_episodes=N_EPISODES, use_extreme_test=False, save_path=None
):
    base = resolve_base()
    xml_path = find_xml()
    run_dir = find_run_dir(base, mode, seed)

    print(f"\n[eval_tracking] Mode: {mode}  Seed: {seed}")
    print(f"[eval_tracking] Run dir: {run_dir}")

    vec_env, env_type = make_tracking_env(
        xml_path, seed, use_extreme_test=use_extreme_test
    )

    # Load VecNormalize
    vn_path = run_dir / "vec_normalize.pkl"
    if vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"[eval_tracking] VecNormalize loaded")
    else:
        print(f"[eval_tracking] WARNING: no vec_normalize.pkl — obs unnormalised")
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    # Check obs space match
    model = SAC.load(str(run_dir / "best_model"), env=vec_env)
    model_dim = model.observation_space.shape[0]
    env_dim = vec_env.observation_space.shape[0]

    if model_dim != env_dim:
        print(f"[eval_tracking] WARNING: obs mismatch model={model_dim} env={env_dim}")
        print(f"[eval_tracking] Rebuilding env to match model obs space")
        vec_env.close()
        # The model may have been trained with a different obs size
        # Load without env to get raw predictions
        model = SAC.load(str(run_dir / "best_model"))
        vec_env, env_type = make_tracking_env(
            xml_path, seed, use_extreme_test=use_extreme_test
        )
        if vn_path.exists():
            vec_env = VecNormalize.load(str(vn_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

    print(f"[eval_tracking] Env type: {env_type}")
    if use_extreme_test:
        print(
            f"[eval_tracking] WARNING: Evaluating under EXTREME zero-shot conditions (200% bounds)!"
        )
    print(f"[eval_tracking] Running {n_episodes} episodes...\n")

    # ── Evaluation loop ──────────────────────────────────────────
    successes, rewards, tracking_errors = [], [], []
    energies, ep_lengths = [], []

    obs = vec_env.reset()
    ep_rew = ep_energy = 0.0
    ep_steps = ep_count = 0
    ep_tracking = []

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec_env.step(action)

        ep_rew += float(rew[0])
        ep_energy += float(np.mean(np.abs(action[0])))
        ep_steps += 1

        # Get tracking error from info if available
        te = info[0].get("tracking_error", info[0].get("goal_dist", float("inf")))
        ep_tracking.append(float(te))

        if done[0]:
            mean_te = float(np.mean(ep_tracking)) if ep_tracking else float("inf")
            # Success: mean tracking error below threshold OR goal reached
            goal_dist = float(info[0].get("goal_dist", mean_te))
            success = (
                mean_te < TRACKING_SUCCESS_THRESHOLD
                or goal_dist < GOAL_SUCCESS_THRESHOLD
            )

            successes.append(float(success))
            rewards.append(ep_rew)
            tracking_errors.append(mean_te)
            energies.append(ep_energy / max(ep_steps, 1))
            ep_lengths.append(ep_steps)

            ep_rew = ep_energy = 0.0
            ep_steps = ep_count_local = 0
            ep_tracking = []
            ep_count += 1

            if ep_count % 10 == 0:
                print(
                    f"  {ep_count}/{n_episodes} | "
                    f"success={np.mean(successes) * 100:.0f}% | "
                    f"mean_track_err={np.mean(tracking_errors):.3f}m"
                )

    vec_env.close()

    results = {
        "mode": mode,
        "seed": seed,
        "env_type": env_type,
        "n_episodes": n_episodes,
        "success_criterion": f"mean_tracking_error < {TRACKING_SUCCESS_THRESHOLD}m",
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_tracking_error": float(np.mean(tracking_errors)),
        "std_tracking_error": float(np.std(tracking_errors)),
        "mean_energy_per_step": float(np.mean(energies)),
        "mean_episode_length": float(np.mean(ep_lengths)),
    }

    print(f"\n{'=' * 55}")
    print(f"  TRACKING EVAL — {mode}  seed={seed}")
    print(f"  Success rate:     {results['success_rate'] * 100:.1f}%")
    print(
        f"  Mean reward:      {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
    )
    print(
        f"  Mean track error: {results['mean_tracking_error']:.3f}m "
        f"± {results['std_tracking_error']:.3f}m"
    )
    print(f"  Energy/step:      {results['mean_energy_per_step']:.4f}")
    print(f"{'=' * 55}")

    if save_path:
        out = Path(save_path)
    else:
        out = run_dir / "tracking_eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval_tracking] Saved → {out}")
    return results


def main():
    global N_EPISODES
    p = argparse.ArgumentParser(description="Evaluate tracking task models")
    p.add_argument(
        "--mode",
        default=None,
        choices=["tracking_none", "tracking_uniform", "tracking_curriculum"],
        help="Training mode (only used if --run-name not provided)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (only used if --run-name not provided)",
    )
    p.add_argument("--all", action="store_true", help="Eval all 3 tracking runs")
    p.add_argument(
        "--episodes", type=int, default=N_EPISODES, help="Number of evaluation episodes"
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for tracking runs (overrides auto-detection)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (e.g., 'tracking_curriculum_cdr_w25_seed0'). Overrides --mode/--seed.",
    )
    p.add_argument(
        "--lookahead",
        type=float,
        nargs="?",
        const=1.0,
        default=None,
        help="Lookahead time in seconds (informational). Use --lookahead alone for default 1.0s, or --lookahead 2.5 for custom value.",
    )
    p.add_argument(
        "--extreme-test",
        action="store_true",
        help="Use 2x out-of-distribution parameters to test sim-to-real proxy bounds.",
    )
    p.add_argument(
        "--save",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Optional explicit path to save the JSON results.",
    )
    p.add_argument(
        "--save-results",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="(Alternative to --save) Explicit path to save the JSON results.",
    )
    args = p.parse_args()

    N_EPISODES = args.episodes

    # Resolve base directory
    if args.base_dir:
        base = Path(args.base_dir)
    else:
        base = resolve_base()

    # Determine save path (--save-results takes precedence over --save)
    # If "auto" const value is used, treat as None for default behavior
    save_path = None
    if args.save_results and args.save_results != "auto":
        save_path = args.save_results
    elif args.save and args.save != "auto":
        save_path = args.save

    if args.run_name:
        # Custom run name provided — evaluate single custom run
        # Mode can be explicitly provided with --mode, or inferred from run_name

        # If --mode is explicitly provided, use it
        # Otherwise try to infer mode from run_name (e.g., tracking_curriculum from tracking_curriculum_cdr_w25_seed0)
        mode = args.mode

        if not mode:
            # Try to infer mode from run_name
            for m in ["tracking_curriculum", "tracking_uniform", "tracking_none"]:
                if m in args.run_name:
                    mode = (
                        "tracking_curriculum"
                        if "curriculum" in m
                        else ("tracking_uniform" if "uniform" in m else "tracking_none")
                    )
                    break

        if not mode:
            print("[eval_tracking] ERROR: Could not infer mode from --run-name.")
            print(
                "[eval_tracking] Provide --mode or ensure run_name contains 'curriculum', 'uniform', or 'none'."
            )
            return

        try:
            run_dir = find_run_dir(base, mode, args.seed, run_name=args.run_name)
            print(f"\n[eval_tracking] Evaluating custom run: {args.run_name}")
            if args.lookahead:
                print(f"[eval_tracking] Lookahead time: {args.lookahead}s")
            evaluate_tracking(
                mode,
                args.seed,
                args.episodes,
                use_extreme_test=args.extreme_test,
                save_path=save_path,
            )
        except FileNotFoundError as e:
            print(f"\n[eval_tracking] ERROR: {e}")
    elif args.all:
        # Evaluate all 3 standard tracking runs
        for mode in ["tracking_none", "tracking_uniform", "tracking_curriculum"]:
            try:
                evaluate_tracking(
                    mode,
                    0,
                    args.episodes,
                    use_extreme_test=args.extreme_test,
                    save_path=save_path,
                )
            except Exception as e:
                print(f"\n[eval_tracking] ERROR {mode}: {e}")
    else:
        # Evaluate single standard run by mode and seed
        if args.mode is None:
            p.error("Provide --mode, --run-name, or --all")
        try:
            evaluate_tracking(
                args.mode,
                args.seed,
                args.episodes,
                use_extreme_test=args.extreme_test,
                save_path=save_path,
            )
        except FileNotFoundError as e:
            print(f"\n[eval_tracking] ERROR: {e}")


if __name__ == "__main__":
    main()
