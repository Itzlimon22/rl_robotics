"""
train_master.py — The "Grand Challenge" Training Script
=======================================================
Combines Trajectory Tracking + Obstacle Avoidance + CDR.
"""

import argparse
import sys
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Ensure local imports work
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "envs"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_obstacle_env import ObstacleAUVWrapper
from auv_dr_wrapper import AUVDomainRandomWrapper, make_training_callbacks
from train import SAC_HYPERPARAMS, detect_device, resolve_save_dir, resolve_xml_path


def make_master_env(xml_path, mode, seed):
    """
    Stacks the Tracking, Obstacle, and DR layers using a Decorator pattern.

    @param {str} xml_path - Path to the MuJoCo model file.
    @param {str} mode - CDR mode (none, uniform, curriculum).
    @param {int} seed - Random seed for reproducibility.
    @return {gym.Env} A wrapped Gymnasium environment.
    """
    # Early Return: Validate path exists
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"XML not found at {xml_path}")

    # 1. Base Task: 3D Figure-8 Tracking
    base = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)

    # 2. Add Hazard: 5 random obstacles + 4 Rangefinder sensors
    obs_env = ObstacleAUVWrapper(base, n_obstacles=5, collision_penalty=20.0)

    # 3. Add Physics Randomization
    dr_env = AUVDomainRandomWrapper(obs_env, mode=mode, seed=seed, verbose=True)

    # 4. Wrap with Monitor to provide accurate episode statistics and prevent UserWarnings
    return Monitor(dr_env)


def train_master(args):
    """
    Handles the end-to-end training pipeline for the AUV agent.

    @param {argparse.Namespace} args - Parsed command line arguments.
    """
    device = detect_device()
    xml = resolve_xml_path(args.xml)
    run_name = f"master_{args.mode}_seed{args.seed}"
    save_dir = resolve_save_dir(f"master_{args.mode}", run_name, args.save_dir)

    def _make():
        return make_master_env(str(xml), args.mode, args.seed)

    # Environment Setup
    train_env = DummyVecEnv([_make])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([_make])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = train_env.obs_rms

    # LOGIC FLOW:
    # 1. Clone the hyperparameters to avoid mutating the original import.
    # 2. Remove 'tensorboard_log' if it exists in the dict to prevent conflicts.
    # 3. Pass the dynamic path explicitly in the constructor.
    custom_params = SAC_HYPERPARAMS.copy()
    custom_params.pop("tensorboard_log", None)

    # Checkpoint Resumption Logic
    checkpoint_path = save_dir / "best_model.zip"
    if checkpoint_path.exists():
        print(f"🔄 Resuming from existing checkpoint: {checkpoint_path}")
        model = SAC.load(str(checkpoint_path), env=train_env, device=device)
        # Ensure tensorboard logging continues in the same directory
        model.tensorboard_log = str(save_dir)
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            device=device,
            seed=args.seed,
            tensorboard_log=str(save_dir),
            **custom_params,
        )

    callbacks = make_training_callbacks(
        eval_env=eval_env,
        log_dir=str(save_dir / "eval"),
        save_path=str(save_dir / "checkpoints"),
    )

    print(f"LAUNCHING MASTER RUN: {run_name} for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)

    # Persistence
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
