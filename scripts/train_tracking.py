"""
train_tracking.py — Train SAC on the 3D trajectory tracking task
================================================================================
Usage:
    python scripts/train_tracking.py --mode curriculum --seed 0 --steps 1000000
"""

import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import utilities from existing train.py (Assuming train.py exists with these helpers)
try:
    from train import (
        SAC_HYPERPARAMS,
        detect_device,
        resolve_save_dir,
        resolve_xml_path,
        build_parser,
        EVAL_FREQ,
        N_EVAL_EPS,
        CHECKPOINT_FREQ,
    )
except ImportError:
    print(
        "Warning: Could not import helpers from train.py. Please ensure train.py is in scripts/"
    )
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, make_training_callbacks


def make_tracking_train_env(
    xml_path: str, mode: str, seed: int, cdr_window_size: int = 50
):
    """Creates the vectorized training environment with DR."""

    def _init():
        env = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)
        env = AUVDomainRandomWrapper(
            env, mode=mode, seed=seed, cdr_window_size=cdr_window_size, verbose=True
        )
        return env

    vec = DummyVecEnv([_init])
    return VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


def make_tracking_eval_env(
    xml_path: str,
    mode: str,
    seed: int,
    train_vn: VecNormalize,
    cdr_window_size: int = 50,
):
    """Creates a deterministic evaluation environment sharing VecNorm stats."""

    def _init():
        env = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)
        env = AUVDomainRandomWrapper(
            env,
            mode=mode,
            seed=seed + 1000,
            cdr_window_size=cdr_window_size,
            verbose=False,
        )
        return env

    vec = DummyVecEnv([_init])
    eval_vn = VecNormalize(
        vec, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99
    )
    eval_vn.obs_rms = train_vn.obs_rms
    eval_vn.ret_rms = train_vn.ret_rms
    eval_vn.training = False
    return eval_vn


def train_tracking(args):
    """Main training loop for the tracking task."""
    t0 = time.time()
    device = detect_device()
    xml = resolve_xml_path(args.xml)

    run_name = args.run_name or f"tracking_{args.mode}_seed{args.seed}"
    save_dir = resolve_save_dir(f"tracking_{args.mode}", run_name, args.save_dir)
    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Tracking Task Training ===")
    print(f"Mode: {args.mode} | Seed: {args.seed} | Steps: {args.steps}")
    print(f"Saving to: {save_dir}")

    train_env = make_tracking_train_env(str(xml), args.mode, args.seed, args.cdr_window)
    eval_env = make_tracking_eval_env(
        str(xml), args.mode, args.seed, train_env, args.cdr_window
    )

    params = dict(SAC_HYPERPARAMS)
    params["tensorboard_log"] = str(tb_dir)

    model = SAC("MlpPolicy", train_env, device=device, seed=args.seed, **params)

    # Use the callback factory from auv_dr_wrapper
    callbacks = make_training_callbacks(
        eval_env=eval_env,
        log_dir=str(save_dir / "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        save_path=str(save_dir / "checkpoints"),
    )

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=run_name,
    )

    model.save(str(save_dir / "best_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min | Saved: {save_dir}")
    train_env.close()
    eval_env.close()


def main():
    p = build_parser()
    p.add_argument(
        "--ablation-param",
        type=str,
        default="none",
        help="Specific physics parameter to randomize (for ablation study)",
    )
    args = p.parse_args()
    train_tracking(args)


if __name__ == "__main__":
    main()
