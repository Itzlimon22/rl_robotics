"""
train_master.py — The "Grand Challenge" Training Script
=======================================================
Combines Trajectory Tracking + Obstacle Avoidance + CDR.
"""

import argparse
import sys
import time
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure local imports work
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "envs"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_obstacle_env import ObstacleAUVWrapper
from auv_dr_wrapper import AUVDomainRandomWrapper, make_training_callbacks
from train import SAC_HYPERPARAMS, detect_device, resolve_save_dir, resolve_xml_path


def make_master_env(xml_path, mode, seed):
    """Stacks the Tracking, Obstacle, and DR layers."""
    # 1. Base Task: 3D Figure-8 Tracking
    base = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)

    # 2. Add Hazard: 5 random obstacles + 4 Rangefinder sensors
    # This expands observation from 19-dim to 23-dim
    obs_env = ObstacleAUVWrapper(base, n_obstacles=5, collision_penalty=20.0)

    # 3. Add Physics Randomization (Curriculum/Uniform/None)
    return AUVDomainRandomWrapper(obs_env, mode=mode, seed=seed, verbose=True)


def train_master(args):
    device = detect_device()
    xml = resolve_xml_path(args.xml)
    run_name = f"master_{args.mode}_seed{args.seed}"
    save_dir = resolve_save_dir(f"master_{args.mode}", run_name, args.save_dir)

    def _make():
        return make_master_env(str(xml), args.mode, args.seed)

    # Vectorize and Normalize
    train_env = DummyVecEnv([_make])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Evaluation Environment (Deterministic)
    eval_env = DummyVecEnv([_make])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = train_env.obs_rms  # Share normalization stats

    model = SAC(
        "MlpPolicy",
        train_env,
        device=device,
        seed=args.seed,
        tensorboard_log=str(run_dir),  # 👈 THIS IS THE FIX
        **SAC_HYPERPARAMS,
    )

    callbacks = make_training_callbacks(
        eval_env=eval_env,
        log_dir=str(save_dir / "eval"),
        save_path=str(save_dir / "checkpoints"),
    )

    print(f"LAUNCHING MASTER RUN: {run_name} for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)

    model.save(str(save_dir / "best_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["none", "uniform", "curriculum"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    train_master(parser.parse_args())
