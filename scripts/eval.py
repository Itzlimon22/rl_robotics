"""
eval.py — Held-out transfer evaluation with energy logging
==========================================================
Loads a trained SAC model and evaluates on TEST_PARAM_CONFIG.
Reports: success rate, mean reward, reward std, energy per step,
         peak thrust, mean final distance.

Usage:
    python scripts/eval.py --mode curriculum --seed 0
    python scripts/eval.py --mode uniform    --seed 1
    python scripts/eval.py --mode none       --seed 2
    python scripts/eval.py --pid
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

N_EVAL_EPISODES = 100


def find_run_dir(mode: str, seed: int, base: Path) -> Path:
    run_dir = base / mode / f"{mode}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def find_xml():
    candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found")


def make_eval_env(xml_path, seed):
    def _init():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=seed + 9999)
        env.set_test_distribution()
        return env

    return DummyVecEnv([_init])


def run_eval_loop(model, vec_env, n_episodes):
    """
    Run evaluation loop. Returns dict of per-episode metrics.
    Works correctly with VecNormalize auto-reset.
    """
    successes, rewards, dists = [], [], []
    energies, peak_thrusts, ep_lengths = [], [], []

    obs = vec_env.reset()
    ep_reward = 0.0
    ep_energy = 0.0
    ep_steps = 0
    ep_peak = 0.0
    ep_count = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Track energy — action is in [-1, 1], mean absolute = effort
        raw_action = action[0]  # unwrap from vec_env batch
        action_mag = float(np.mean(np.abs(raw_action)))
        ep_energy += action_mag
        ep_peak = max(ep_peak, float(np.max(np.abs(raw_action))))
        ep_reward += float(reward[0])
        ep_steps += 1

        if done[0]:
            goal_dist = float(info[0].get("goal_dist", float("inf")))
            success = goal_dist < 0.5

            successes.append(float(success))
            rewards.append(ep_reward)
            dists.append(goal_dist)
            energies.append(ep_energy / max(ep_steps, 1))
            peak_thrusts.append(ep_peak)
            ep_lengths.append(ep_steps)

            ep_count += 1
            ep_reward = 0.0
            ep_energy = 0.0
            ep_steps = 0
            ep_peak = 0.0

            if ep_count % 10 == 0:
                print(
                    f"  {ep_count}/{n_episodes} | "
                    f"success={np.mean(successes) * 100:.0f}% | "
                    f"energy={np.mean(energies):.3f}"
                )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_dist": float(np.mean(dists)),
        "std_dist": float(np.std(dists)),
        "mean_energy_per_step": float(np.mean(energies)),
        "std_energy_per_step": float(np.std(energies)),
        "mean_peak_thrust": float(np.mean(peak_thrusts)),
        "mean_episode_length": float(np.mean(ep_lengths)),
        "n_episodes": n_episodes,
    }


def evaluate_rl(args):
    on_colab = Path("/content/drive/MyDrive").exists()
    base = (
        Path("/content/drive/MyDrive/rl_research/auv")
        if on_colab
        else Path.home() / "rl_research" / "auv"
    )

    run_dir = find_run_dir(args.mode, args.seed, base)
    xml_path = find_xml()

    print(f"[eval] Loading: {run_dir}")
    print(f"[eval] Test distribution: drag_lateral=[0.30,0.80], current=[0.20,0.60]")

    vec_env = make_eval_env(xml_path, args.seed)
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)

    print(f"\n[eval] Running {N_EVAL_EPISODES} episodes...\n")
    results = run_eval_loop(model, vec_env, N_EVAL_EPISODES)
    results["mode"] = args.mode
    results["seed"] = args.seed
    results["eval_type"] = "held_out_test"

    print(f"\n{'=' * 55}")
    print(f"  TRANSFER EVAL — {args.mode.upper()} seed={args.seed}")
    print(f"  Success rate:    {results['success_rate'] * 100:.1f}%")
    print(
        f"  Mean reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
    )
    print(
        f"  Mean dist:       {results['mean_dist']:.2f}m ± {results['std_dist']:.2f}m"
    )
    print(
        f"  Energy/step:     {results['mean_energy_per_step']:.4f} ± {results['std_energy_per_step']:.4f}"
    )
    print(f"  Peak thrust:     {results['mean_peak_thrust']:.4f}")
    print(f"  Mean ep length:  {results['mean_episode_length']:.1f} steps")
    print(f"{'=' * 55}\n")

    out = run_dir / "test_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved → {out}")
    vec_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pid", action="store_true")
    args = p.parse_args()

    if args.pid:
        print("Run: python scripts/pid_baseline.py --episodes 100 --test-dist")
    else:
        evaluate_rl(args)


if __name__ == "__main__":
    main()
