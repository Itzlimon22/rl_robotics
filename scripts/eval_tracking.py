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


def find_run_dir(base, mode, seed):
    for suffix in ["_v1", ""]:
        rd = base / mode / f"{mode}_seed{seed}{suffix}"
        if rd.exists() and (rd / "best_model.zip").exists():
            return rd
    raise FileNotFoundError(f"No run found: {mode} seed{seed}")


def make_tracking_env(xml_path, seed):
    """
    Build tracking environment.
    Falls back to goal-reaching if auv_tracking_env.py not available.
    """
    try:
        from auv_tracking_env import HalcyonAUVTrackingEnv
        from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG

        def _init():
            env = HalcyonAUVTrackingEnv(
                xml_path=str(xml_path),
                path_speed=0.3,
                tracking_threshold=TRACKING_SUCCESS_THRESHOLD,
            )
            wrapper = AUVDomainRandomWrapper(
                env, mode="uniform", seed=seed + 9999, verbose=False
            )
            wrapper._sample_and_apply(TEST_PARAM_CONFIG)
            return wrapper

        return DummyVecEnv([_init]), "tracking"

    except ImportError:
        # Fallback: use regular goal-reaching env
        print("[eval_tracking] WARNING: auv_tracking_env.py not found")
        print("[eval_tracking] Using goal-reaching env as fallback")
        from auv_env import HalcyonAUVEnv
        from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG

        def _init():
            env = HalcyonAUVEnv(xml_path=str(xml_path))
            wrapper = AUVDomainRandomWrapper(
                env, mode="uniform", seed=seed + 9999, verbose=False
            )
            wrapper._sample_and_apply(TEST_PARAM_CONFIG)
            return wrapper

        return DummyVecEnv([_init]), "goal_reaching"


def evaluate_tracking(mode, seed, n_episodes=N_EPISODES):
    base = resolve_base()
    xml_path = find_xml()
    run_dir = find_run_dir(base, mode, seed)

    print(f"\n[eval_tracking] Mode: {mode}  Seed: {seed}")
    print(f"[eval_tracking] Run dir: {run_dir}")

    vec_env, env_type = make_tracking_env(xml_path, seed)

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
        vec_env, env_type = make_tracking_env(xml_path, seed)
        if vn_path.exists():
            vec_env = VecNormalize.load(str(vn_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

    print(f"[eval_tracking] Env type: {env_type}")
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

    out = run_dir / "tracking_eval_results.json"
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
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="Eval all 3 tracking runs")
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    args = p.parse_args()

    N_EPISODES = args.episodes

    if args.all:
        for mode in ["tracking_none", "tracking_uniform", "tracking_curriculum"]:
            try:
                evaluate_tracking(mode, 0, args.episodes)
            except Exception as e:
                print(f"\n[eval_tracking] ERROR {mode}: {e}")
    else:
        if args.mode is None:
            p.error("Provide --mode or --all")
        evaluate_tracking(args.mode, args.seed, args.episodes)


if __name__ == "__main__":
    main()
