"""
eval_transfer.py — Zero-shot cross-model transfer evaluation
=============================================================
Loads a policy trained on auv.xml and evaluates it on auv_transfer.xml
WITHOUT any retraining. This tests whether CDR generalises to a
different physical AUV model (different mass, drag, geometry).

Usage (Mac):
    cd ~/rl_robotics && conda activate rl
    python scripts/eval_transfer.py --source-mode curriculum --source-seed 1 --episodes 50

Usage (Colab):
    !python /content/rl_robotics/scripts/eval_transfer.py \
        --source-mode curriculum --source-seed 1 --episodes 50
"""

import argparse, json, sys
import numpy as np
from pathlib import Path

_REPO = Path(__file__).parent.parent.resolve()
for p in [_REPO, _REPO / "envs"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG
import mujoco, gymnasium as gym
from gymnasium import spaces

# ── Transfer physics — applied ON TOP of the new XML ─────────
# These represent the 25% drag increase for the transfer model
TRANSFER_PHYSICS_DELTA = {
    "c_drag_lateral": 1.25,  # multiplier (25% more drag)
    "c_drag_axial": 1.25,  # multiplier
}

# ── Test distribution (same as original eval) ─────────────────
TEST_CONFIG = {
    "c_drag_lateral": (0.30 * 1.25, 0.80 * 1.25),  # scaled for transfer AUV
    "c_drag_axial": (0.12 * 1.25, 0.32 * 1.25),
    "buoyancy_offset": (-0.10, 0.15),
    "current_speed": (0.20, 0.60),
    "added_mass": (0.20, 0.50),
    "act_efficiency": (0.70, 1.00),
}

ALL_MODES = ["none", "uniform", "curriculum"]


def find_run(base, mode, seed):
    candidates = [
        base / mode / f"{mode}_seed{seed}",
        base / f"master_{mode}" / f"master_{mode}_seed{seed}",
    ]
    for c in candidates:
        if c.exists() and (c / "best_model.zip").exists():
            return c
    raise FileNotFoundError(f"No model for {mode} seed{seed} in {base}")


def make_transfer_env(xml_path, dr_mode, seed):
    """Create env using transfer XML (different AUV body)."""
    from auv_env import HalcyonAUVEnv

    base_env = HalcyonAUVEnv(xml_path=xml_path)
    # Apply additional drag multiplier to simulate heavier AUV
    base_env.physics_params["c_drag_lateral"] *= TRANSFER_PHYSICS_DELTA[
        "c_drag_lateral"
    ]
    base_env.physics_params["c_drag_axial"] *= TRANSFER_PHYSICS_DELTA["c_drag_axial"]
    wrapped = AUVDomainRandomWrapper(base_env, mode=dr_mode, seed=seed + 777)
    wrapped._test_mode = True  # Force test distribution
    return wrapped


def run_eval(model, vec_env, n_episodes):
    successes, rewards, dists, energies, lengths = [], [], [], [], []
    obs = vec_env.reset()
    ep_rew = ep_energy = ep_peak = 0.0
    ep_steps = ep_count = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = vec_env.step(action)
        a0 = action[0]
        ep_energy += float(np.mean(np.abs(a0)))
        ep_rew += float(rew[0])
        ep_steps += 1
        if done[0]:
            gd = float(info[0].get("goal_dist", 999.0))
            successes.append(float(gd < 0.5))
            rewards.append(ep_rew)
            dists.append(gd)
            energies.append(ep_energy / max(ep_steps, 1))
            lengths.append(ep_steps)
            ep_count += 1
            ep_rew = ep_energy = 0.0
            ep_steps = 0
            if ep_count % 10 == 0:
                print(
                    f"    {ep_count}/{n_episodes} | "
                    f"success={np.mean(successes) * 100:.0f}% | "
                    f"energy={np.mean(energies):.4f}"
                )
    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_dist": float(np.mean(dists)),
        "mean_energy": float(np.mean(energies)),
        "n_episodes": n_episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-mode", default="curriculum")
    p.add_argument("--source-seed", type=int, default=1)
    p.add_argument(
        "--target-xml",
        default=None,
        help="Path to transfer AUV XML. Default: envs/auv_transfer.xml",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--all-modes",
        action="store_true",
        help="Eval all three modes for comparison table",
    )
    p.add_argument("--save", default=None)
    args = p.parse_args()

    on_colab = Path("/content/drive/MyDrive").exists()
    base = (
        Path("/content/drive/MyDrive/rl_research/auv")
        if on_colab
        else Path.home() / "rl_research" / "auv"
    )

    transfer_xml = args.target_xml or str(_REPO / "envs" / "auv_transfer.xml")
    if not Path(transfer_xml).exists():
        raise FileNotFoundError(
            f"Transfer XML not found: {transfer_xml}\n"
            "Create envs/auv_transfer.xml first (see A10a Part 2.2)"
        )

    modes_to_eval = ALL_MODES if args.all_modes else [args.source_mode]
    seeds_to_eval = [0, 1, 2] if args.all_modes else [args.source_seed]

    print(f"\n{'=' * 60}")
    print(f"A10a: Zero-Shot Cross-Model Transfer Evaluation")
    print(f"Transfer XML: {transfer_xml}")
    print(f"Episodes:     {args.episodes}")
    print(f"{'=' * 60}\n")

    all_results = {}

    for mode in modes_to_eval:
        all_results[mode] = []
        for seed in seeds_to_eval:
            print(f"\n── {mode} seed={seed} ──")
            try:
                run_dir = find_run(base, mode, seed)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            dr_mode = mode.replace("master_", "")

            def _make():
                return make_transfer_env(transfer_xml, dr_mode, seed)

            vec_env = DummyVecEnv([_make])
            vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            model = SAC.load(str(run_dir / "best_model"), env=vec_env)

            result = run_eval(model, vec_env, args.episodes)
            result["mode"] = mode
            result["seed"] = seed
            result["transfer_xml"] = transfer_xml
            all_results[mode].append(result)

            print(
                f"  ✓ {mode} seed={seed}: "
                f"success={result['success_rate'] * 100:.1f}% | "
                f"reward={result['mean_reward']:.1f} | "
                f"energy={result['mean_energy']:.4f}"
            )
            vec_env.close()

    # Aggregate and print summary
    print(f"\n{'=' * 60}")
    print("TRANSFER RESULTS (mean across seeds)")
    print(f"{'=' * 60}")
    print(f"{'Mode':<22} {'Success%':>10} {'Reward':>10} {'Energy':>10}")
    print("-" * 55)

    labels = {"none": "Naive SAC", "uniform": "Uniform DR", "curriculum": "CDR (Ours)"}
    for mode in modes_to_eval:
        results = all_results[mode]
        if not results:
            continue
        sr = np.mean([r["success_rate"] for r in results]) * 100
        rw = np.mean([r["mean_reward"] for r in results])
        en = np.mean([r["mean_energy"] for r in results])
        print(f"{labels.get(mode, mode):<22} {sr:>8.1f}%  {rw:>9.1f}  {en:>9.4f}")

    # Save
    save_path = args.save or str(base / "transfer_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
