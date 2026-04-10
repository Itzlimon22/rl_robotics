"""
eval_ablation.py — Evaluate all 15 ablation models on held-out test distribution
=================================================================================
Loads each ablation model (trained with only ONE parameter randomised)
and evaluates it on the FULL test distribution (all parameters shifted).

This reveals which single parameter provides the most transfer benefit
when randomised alone.

Usage:
    python scripts/eval_ablation.py                     # eval all 15
    python scripts/eval_ablation.py --param c_drag_lateral  # one param, all seeds
    python scripts/eval_ablation.py --param current_speed --seed 0

Output:
    ~/rl_research/auv/ablation_<param>/ablation_<param>_<seed>/ablation_eval_results.json
    ~/rl_research/auv/ablation_summary.json   ← aggregated table
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
from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG

# ── Constants ─────────────────────────────────────────────────────────────────

N_EPISODES = 50  # 50 episodes per ablation model (faster than 100)

# Mapping from friendly name → folder prefix
PARAM_FOLDERS = {
    "c_drag_lateral": "ablation_cdraglateral",
    "c_drag_axial": "ablation_cdragaxial",
    "buoyancy_offset": "ablation_buoyancyoffset",
    "current_speed": "ablation_currentspeed",
    "added_mass": "ablation_addedmass",
    "act_efficiency": "ablation_actefficiency",
}

# Friendly display names for tables/figures
PARAM_LABELS = {
    "c_drag_lateral": "Lateral drag",
    "c_drag_axial": "Axial drag",
    "buoyancy_offset": "Buoyancy",
    "current_speed": "Water current",
    "added_mass": "Added mass",
    "act_efficiency": "Act. efficiency",
}


def get_base_dir() -> Path:
    colab = Path("/content/drive/MyDrive/rl_research/auv")
    local = Path.home() / "rl_research" / "auv"
    return colab if colab.exists() else local


def find_xml() -> Path:
    for p in [
        _ENVS_DIR / "auv.xml",
        _REPO_ROOT / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("/content/rl_robotics/envs/auv.xml"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found")


def find_ablation_run_dir(base: Path, param: str, seed: int) -> Path:
    """Find run directory for ablation_{param}_{seed}."""
    folder = PARAM_FOLDERS[param]
    run_dir = base / folder / f"{folder}_{seed}"
    if run_dir.exists() and (run_dir / "best_model.zip").exists():
        return run_dir
    raise FileNotFoundError(
        f"Ablation run not found: {run_dir}\n"
        f"Train first: python scripts/train.py --mode uniform --seed {seed} "
        f"--ablation-param {param} --steps 500000"
    )


def evaluate_ablation_model(
    param: str,
    seed: int,
    base: Path,
    xml_path: Path,
    n_episodes: int = N_EPISODES,
) -> dict:
    """
    Load ablation model (trained with only `param` randomised)
    and evaluate on FULL test distribution (all 6 parameters shifted).
    """
    run_dir = find_ablation_run_dir(base, param, seed)
    print(f"\n[eval] {PARAM_LABELS[param]:18s} seed{seed} → {run_dir.name}")

    # Build FULL test distribution env (all parameters at test ranges)
    def _init():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        wrapper = AUVDomainRandomWrapper(
            env, mode="uniform", seed=seed + 99999, verbose=False
        )
        # Apply full test distribution — all 6 parameters hard
        wrapper._sample_and_apply(TEST_PARAM_CONFIG)
        return wrapper

    vec_env = DummyVecEnv([_init])

    # Load VecNormalize stats from training
    vn_path = run_dir / "vec_normalize.pkl"
    if vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"  WARNING: no vec_normalize.pkl — obs unnormalised")

    # Load model
    model = SAC.load(str(run_dir / "best_model"), env=vec_env)

    # Evaluation loop
    successes, rewards, dists, energies, ep_lengths = [], [], [], [], []
    obs = vec_env.reset()
    ep_rew = ep_energy = 0.0
    ep_steps = ep_count = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec_env.step(action)

        ep_rew += float(rew[0])
        ep_energy += float(np.mean(np.abs(action[0])))
        ep_steps += 1

        if done[0]:
            gd = float(info[0].get("goal_dist", float("inf")))
            successes.append(float(gd < 0.5))
            rewards.append(ep_rew)
            dists.append(gd)
            energies.append(ep_energy / max(ep_steps, 1))
            ep_lengths.append(ep_steps)
            ep_rew = ep_energy = 0.0
            ep_steps = 0
            ep_count += 1

    vec_env.close()

    results = {
        "param": param,
        "param_label": PARAM_LABELS[param],
        "seed": seed,
        "run_dir": str(run_dir),
        "n_episodes": n_episodes,
        "eval_distribution": "full_test",
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_dist": float(np.mean(dists)),
        "mean_energy": float(np.mean(energies)),
        "mean_ep_length": float(np.mean(ep_lengths)),
    }

    print(
        f"  → success={results['success_rate'] * 100:.1f}%  "
        f"dist={results['mean_dist']:.2f}m  "
        f"energy={results['mean_energy']:.4f}"
    )

    # Save per-run JSON
    out = run_dir / "ablation_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_and_print(all_results: list, base: Path):
    """Aggregate results across seeds and print summary table."""
    from collections import defaultdict

    by_param = defaultdict(list)
    for r in all_results:
        by_param[r["param"]].append(r)

    print(f"\n{'=' * 75}")
    print(f"  ABLATION RESULTS — Full Test Distribution")
    print(
        f"  (Compare to Uniform DR: {'{FILL:sr_udr}'}% | No DR: {'{FILL:sr_naive}'}%)"
    )
    print(f"{'=' * 75}")
    print(f"  {'Parameter':<22} {'Success (%)':>14} {'Dist (m)':>10} {'Energy':>10}")
    print(f"  {'-' * 22} {'-' * 14} {'-' * 10} {'-' * 10}")

    summary = []
    for param in PARAM_FOLDERS:
        results = by_param.get(param, [])
        if not results:
            print(f"  {PARAM_LABELS[param]:<22} {'NO DATA':>14}")
            continue

        srs = [r["success_rate"] * 100 for r in results]
        dsts = [r["mean_dist"] for r in results]
        engs = [r["mean_energy"] for r in results]

        mu_sr, sd_sr = np.mean(srs), np.std(srs)
        mu_dst, sd_dst = np.mean(dsts), np.std(dsts)
        mu_eng = np.mean(engs)

        print(
            f"  {PARAM_LABELS[param]:<22} "
            f"{mu_sr:>7.1f}±{sd_sr:<6.1f}% "
            f"{mu_dst:>7.2f}m "
            f"{mu_eng:>10.4f}"
        )

        summary.append(
            {
                "param": param,
                "label": PARAM_LABELS[param],
                "sr_mean": float(mu_sr),
                "sr_std": float(sd_sr),
                "dist_mean": float(mu_dst),
                "energy_mean": float(mu_eng),
                "n_seeds": len(results),
            }
        )

    print(f"{'=' * 75}\n")

    # Save aggregate summary
    out_path = base / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Summary saved → {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Evaluate all ablation models on held-out test distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--param",
        type=str,
        default=None,
        choices=list(PARAM_FOLDERS.keys()),
        help="Evaluate only this parameter (all seeds). Default: all 5.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluate only this seed. Default: all 3.",
    )
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    args = p.parse_args()

    base = get_base_dir()
    xml_path = find_xml()

    # Determine what to evaluate
    params = [args.param] if args.param else list(PARAM_FOLDERS.keys())
    seeds = [args.seed] if args.seed is not None else [0, 1, 2]

    print(f"[eval] Base dir: {base}")
    print(f"[eval] Params:   {params}")
    print(f"[eval] Seeds:    {seeds}")
    print(f"[eval] Episodes: {args.episodes}\n")

    all_results = []
    for param in params:
        for seed in seeds:
            try:
                r = evaluate_ablation_model(param, seed, base, xml_path, args.episodes)
                all_results.append(r)
            except FileNotFoundError as e:
                print(f"  SKIP — {e}")
            except Exception as e:
                print(f"  ERROR {param} seed{seed}: {e}")
                import traceback

                traceback.print_exc()

    if len(all_results) > 1:
        aggregate_and_print(all_results, base)


if __name__ == "__main__":
    main()
