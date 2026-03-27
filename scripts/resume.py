"""
resume.py — Resume interrupted AUV training from checkpoint
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
# Insert _SCRIPT_DIR before _ENVS_DIR to prioritize scripts/train.py
for _p in [_REPO_ROOT, _SCRIPT_DIR, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

from train import (
    SAC_HYPERPARAMS,
    EVAL_FREQ,
    N_EVAL_EPS,
    CHECKPOINT_FREQ,
    detect_device,
    resolve_xml_path,
    AUVMetricsCallback,
    CDRCheckpointCallback,
)


def find_run_dir(mode, seed, base_dir=None):
    run_name = f"{mode}_seed{seed}"
    candidates = []
    if base_dir:
        candidates.append(Path(base_dir) / mode / run_name)
    candidates += [
        Path("/content/drive/MyDrive/rl_research/auv") / mode / run_name,
        Path.home() / "rl_research" / "auv" / mode / run_name,
    ]
    for p in candidates:
        if p.exists():
            print(f"[resume] Run dir: {p}")
            return p
    raise FileNotFoundError(
        f"Run dir not found for mode={mode} seed={seed}. Tried: {candidates}"
    )


def find_checkpoint_files(run_dir, checkpoint_step):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ in {run_dir}")

    model_path = ckpt_dir / f"model_{checkpoint_step}_steps.zip"
    if not model_path.exists():
        available = sorted(
            [
                int(p.stem.replace("model_", "").replace("_steps", ""))
                for p in ckpt_dir.glob("model_*_steps.zip")
            ]
        )
        raise FileNotFoundError(
            f"model_{checkpoint_step}_steps.zip not found. Available: {available}"
        )

    # Find best vecnormalize at or before checkpoint_step
    vn_files = sorted(ckpt_dir.glob("vecnormalize_*_steps.pkl"))
    if not vn_files:
        # VecNormalize was not saved (SB3 version issue) — return None
        # rebuild_train_env will reconstruct stats via warm-up rollout
        print("[resume] WARNING: No vecnormalize_*.pkl found in checkpoints/")
        print(
            "[resume] Will reconstruct VecNormalize stats via warm-up rollout (~5k steps)"
        )
        return model_path, None

    vn_steps = [
        int(p.stem.replace("vecnormalize_", "").replace("_steps", "")) for p in vn_files
    ]
    valid = [s for s in vn_steps if s <= checkpoint_step]
    if not valid:
        print(
            f"[resume] WARNING: No vecnormalize <= {checkpoint_step}. Reconstructing via warm-up."
        )
        return model_path, None

    best_vn = max(valid)
    vecnorm_path = ckpt_dir / f"vecnormalize_{best_vn}_steps.pkl"
    if best_vn != checkpoint_step:
        print(
            f"[resume] VecNormalize: using step {best_vn} (closest to {checkpoint_step})"
        )
    return model_path, vecnorm_path


def find_cdr_state(run_dir, checkpoint_step):
    cdr_files = [
        f
        for f in sorted(run_dir.glob("cdr_state_*.json"))
        if f.stem.replace("cdr_state_", "").isdigit()
    ]
    if not cdr_files:
        print("[resume] No CDR state found — curriculum starts fresh")
        return None
    steps = [int(f.stem.replace("cdr_state_", "")) for f in cdr_files]
    valid = [s for s in steps if s <= checkpoint_step]
    if not valid:
        return None
    best = max(valid)
    with open(run_dir / f"cdr_state_{best}.json") as f:
        state = json.load(f)
    print(
        f"[resume] CDR state from step {best}: level={state['curriculum_level']:.3f} "
        f"episodes={state['n_episodes']} successes={state['n_successes']}"
    )
    return state


def rebuild_train_env(xml_path, mode, seed, vecnorm_path, cdr_state):
    def _init():
        env = HalcyonAUVEnv(xml_path=xml_path)
        wrapper = AUVDomainRandomWrapper(env, mode=mode, seed=seed, verbose=True)
        if cdr_state and mode == "curriculum":
            wrapper.load_cdr_state(cdr_state)
            print(f"[resume] CDR restored: level={wrapper.curriculum_level:.3f}")
        return wrapper

    vec = DummyVecEnv([_init])

    if vecnorm_path is not None:
        # Load saved VecNormalize stats
        vec = VecNormalize.load(str(vecnorm_path), vec)
        print(f"[resume] VecNormalize loaded from {vecnorm_path.name}")
    else:
        # Reconstruct stats via warm-up rollout
        # Create fresh VecNormalize and run 5k random steps to approximate stats
        print("[resume] Reconstructing VecNormalize via 5,000-step warm-up...")
        vec = VecNormalize(
            vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=SAC_HYPERPARAMS["gamma"],
        )
        obs = vec.reset()
        for i in range(5000):
            action = np.array([vec.action_space.sample()])
            obs, _, done, _ = vec.step(action)
            if done[0]:
                obs = vec.reset()
            if (i + 1) % 1000 == 0:
                print(f"  warm-up {i + 1}/5000 steps...")
        print("[resume] Warm-up complete. Stats reconstructed.")

    vec.training = True
    vec.norm_reward = True
    return vec


def rebuild_eval_env(xml_path, mode, seed, train_vec):
    def _init():
        env = HalcyonAUVEnv(xml_path=xml_path)
        return AUVDomainRandomWrapper(env, mode=mode, seed=seed + 1000, verbose=False)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=SAC_HYPERPARAMS["gamma"],
    )
    vec.obs_rms = train_vec.obs_rms
    vec.ret_rms = train_vec.ret_rms
    vec.training = False
    return vec


def resume(args):
    t_start = time.time()
    device = detect_device()
    xml_path = resolve_xml_path(args.xml)
    run_dir = find_run_dir(args.mode, args.seed, args.save_dir)
    model_path, vecnorm_path = find_checkpoint_files(run_dir, args.checkpoint)
    cdr_state = find_cdr_state(run_dir, args.checkpoint)

    remaining = args.total_steps - args.checkpoint
    if remaining <= 0:
        print(f"Already at {args.total_steps} steps. Nothing to do.")
        return

    print(f"\n{'=' * 60}")
    print(f"  RESUMING mode={args.mode} seed={args.seed}")
    print(f"  {args.checkpoint:,} → {args.total_steps:,}  ({remaining:,} remaining)")
    print(f"  Model:   {model_path.name}")
    print(f"  VecNorm: {vecnorm_path.name}")
    print(f"{'=' * 60}\n")

    train_env = rebuild_train_env(
        str(xml_path), args.mode, args.seed, vecnorm_path, cdr_state
    )
    eval_env = rebuild_eval_env(str(xml_path), args.mode, args.seed, train_env)

    model = SAC.load(str(model_path), env=train_env, device=device)
    model.tensorboard_log = str(run_dir / "tensorboard")
    print(f"[model] Loaded. Buffer size: {model.replay_buffer.size()}")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    callbacks = CallbackList(
        [
            AUVMetricsCallback(),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPS,
                deterministic=True,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=CHECKPOINT_FREQ,
                save_path=str(ckpt_dir),
                name_prefix="model",
                save_vecnormalize=True,
                verbose=1,
            ),
            CDRCheckpointCallback(
                save_dir=run_dir, save_freq=CHECKPOINT_FREQ, verbose=1
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,  # CRITICAL: keeps step counter continuous
            tb_log_name=f"{args.mode}_seed{args.seed}",
        )
    except KeyboardInterrupt:
        print("\n[resume] Interrupted — saving...")

    # Save final artifacts
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vec_normalize.pkl"))
    try:
        env = train_env.venv.envs[0]
        if hasattr(env, "get_cdr_state"):
            state = env.get_cdr_state()
            with open(run_dir / "cdr_state.json", "w") as f:
                json.dump(state, f, indent=2)
            print(
                f"[save] CDR final: level={state['curriculum_level']:.3f} "
                f"episodes={state['n_episodes']} successes={state['n_successes']}"
            )
    except Exception as e:
        print(f"[save] CDR state warning: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE — {remaining:,} steps in {elapsed / 60:.1f} min")
    print(f"  Best model: {run_dir}/best_model.zip")
    print(f"{'=' * 60}")
    train_env.close()
    eval_env.close()


def main():
    p = argparse.ArgumentParser(
        description="Resume interrupted AUV training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=int,
        help="Step number to resume from, e.g. 400000",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="Original target. Remaining = total_steps - checkpoint.",
    )
    p.add_argument("--xml", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    args = p.parse_args()
    resume(args)


if __name__ == "__main__":
    main()
