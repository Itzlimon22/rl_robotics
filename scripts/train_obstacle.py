"""
train_obstacle.py — Train SAC on obstacle avoidance task
========================================================
Same setup as train.py but uses ObstacleAUVWrapper.
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

from train import (
    SAC_HYPERPARAMS,
    detect_device,
    resolve_save_dir,
    resolve_xml_path,
    AUVMetricsCallback,
    CDRCheckpointCallback,
    build_parser,
    EVAL_FREQ,
    N_EVAL_EPS,
    CHECKPOINT_FREQ,
)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.auv_dr_wrapper import make_obstacle_env


def make_obs_train_env(xml_path, mode, seed):
    def _init():
        return make_obstacle_env(xml_path, mode=mode, seed=seed, n_obstacles=5)

    vec = DummyVecEnv([_init])
    return VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


def make_obs_eval_env(xml_path, mode, seed, train_vn):
    def _init():
        return make_obstacle_env(xml_path, mode=mode, seed=seed + 1000, n_obstacles=5)

    vec = DummyVecEnv([_init])
    eval_vn = VecNormalize(
        vec, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99
    )
    eval_vn.obs_rms = train_vn.obs_rms
    eval_vn.ret_rms = train_vn.ret_rms
    eval_vn.training = False
    return eval_vn


def train_obstacle(args):
    t0 = time.time()
    device = detect_device()
    xml = resolve_xml_path(args.xml)

    run_name = args.run_name or f"obstacle_{args.mode}_seed{args.seed}"
    save_dir = resolve_save_dir(f"obstacle_{args.mode}", run_name, args.save_dir)
    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_obs_train_env(str(xml), args.mode, args.seed)
    eval_env = make_obs_eval_env(str(xml), args.mode, args.seed, train_env)

    params = dict(SAC_HYPERPARAMS)
    params["tensorboard_log"] = str(tb_dir)

    model = SAC("MlpPolicy", train_env, device=device, seed=args.seed, **params)

    callbacks = CallbackList(
        [
            AUVMetricsCallback(),
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(save_dir / "eval"),
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPS,
                deterministic=True,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=CHECKPOINT_FREQ,
                save_path=str(save_dir / "checkpoints"),
                name_prefix="obstacle_model",
                save_vecnormalize=True,
            ),
            CDRCheckpointCallback(save_dir=save_dir, save_freq=CHECKPOINT_FREQ),
        ]
    )

    print(f"\nStarting Obstacle Avoidance Training: {args.mode} | Seed {args.seed}")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=run_name,
    )

    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min | Saved: {save_dir}")

    train_env.close()
    eval_env.close()


def main():
    p = build_parser()
    args = p.parse_args()
    train_obstacle(args)


if __name__ == "__main__":
    main()
