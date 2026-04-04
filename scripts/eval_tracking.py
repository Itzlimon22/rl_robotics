"""
eval_tracking.py — Evaluate trajectory tracking models
=======================================================
Loads a trained tracking model and evaluates on held-out
test physics distribution.

Reports: mean tracking error, path progress, reward, energy.

Usage:
    python scripts/eval_tracking.py --mode curriculum --seed 0
    python scripts/eval_tracking.py --mode uniform --seed 0
    python scripts/eval_tracking.py --mode none --seed 0
"""

from __future__ import annotations
import argparse
import json
import os
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
from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG
from auv_obstacle_env import ObstacleAUVWrapper
import gymnasium as gym
from gymnasium import spaces


class ObsShapeAdapter(gym.Wrapper):
    """Adapts observation shape by padding or truncating to match expected dimensions."""

    def __init__(self, env, target_shape):
        super().__init__(env)
        self.target_shape = target_shape
        current_shape = env.observation_space.shape[0]
        self.pad_amount = max(0, target_shape - current_shape)
        self.truncate_amount = max(0, current_shape - target_shape)

        # Update observation space
        low = np.concatenate(
            [
                np.array(env.observation_space.low),
                np.full(self.pad_amount, -np.inf),
            ]
        )[:target_shape]
        high = np.concatenate(
            [
                np.array(env.observation_space.high),
                np.full(self.pad_amount, np.inf),
            ]
        )[:target_shape]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._adapt_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._adapt_obs(obs)
        return obs, info

    def _adapt_obs(self, obs):
        """Pad or truncate observation to target shape."""
        current_len = obs.shape[0] if isinstance(obs, np.ndarray) else len(obs)
        if self.pad_amount > 0:
            obs = np.concatenate([obs, np.zeros(self.pad_amount, dtype=np.float32)])
        elif self.truncate_amount > 0:
            obs = obs[: self.target_shape]
        return obs.astype(np.float32)


def resolve_paths(mode, seed):
    on_colab = os.path.exists("/content/drive/MyDrive")
    if on_colab:
        base = Path("/content/drive/MyDrive/rl_research/auv")
    else:
        base = Path.home() / "rl_research" / "auv"

    run_name = f"tracking_{mode}_seed{seed}"
    run_dir = base / f"tracking_{mode}" / run_name

    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("/content/rl_robotics/envs/auv.xml"),
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        raise FileNotFoundError("auv.xml not found")

    return run_dir, xml_path


def evaluate_tracking(mode, seed, n_episodes=50, use_obstacle=False):
    run_dir, xml_path = resolve_paths(mode, seed)

    model_path = run_dir / "best_model.zip"
    vecnorm_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    print(f"\n[tracking_eval] Mode={mode} Seed={seed}")
    if use_obstacle:
        print(f"[tracking_eval] Using obstacle wrapper")
    print(f"[tracking_eval] Loading from: {run_dir}")

    def make_env():
        # Tracking env with test physics distribution
        env = HalcyonAUVTrackingEnv(
            xml_path=str(xml_path),
            path_speed=0.3,
            tracking_threshold=1.0,
            max_tracking_error=5.0,
        )
        wrapper = AUVDomainRandomWrapper(
            env, mode=mode, seed=seed + 9999, verbose=False
        )
        wrapper.set_test_distribution()

        # Inject obstacle wrapper if flag is passed
        if use_obstacle:
            wrapper = ObstacleAUVWrapper(wrapper)

        return wrapper

    vec_env = DummyVecEnv([make_env])

    # Try loading VecNormalize; if shape mismatch, load model without normalization
    model_env = None
    try:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        model_env = vec_env
        print(f"[tracking_eval] VecNormalize loaded successfully")
    except AssertionError as e:
        if "spaces must have the same shape" in str(e):
            print(f"[tracking_eval] Shape mismatch: {e}")
            print(
                f"[tracking_eval] Loading model without VecNormalize normalization..."
            )
            # Use raw observations - don't load vecnorm
            model_env = None
        else:
            raise

    if model_env is None:
        vec_env.training = False
        vec_env.norm_reward = False
        model = SAC.load(str(model_path))
        print(f"[tracking_eval] ✓ Model loaded (no VecNormalize)")

        # Check if observation shape mismatch and wrap if needed
        model_obs_shape = model.policy.observation_space.shape[0]
        env_obs_shape = vec_env.observation_space.shape[0]
        if model_obs_shape != env_obs_shape:
            print(
                f"[tracking_eval] Obs shape mismatch: model expects {model_obs_shape}, "
                f"env provides {env_obs_shape}. Padding observations..."
            )

            # Wrap vec_env to adapt observations
            # Create a wrapper around the vectorized environment
            class VecObsAdapter:
                """Adapts vectorized observations by padding."""

                def __init__(self, vec_env, target_shape):
                    self.vec_env = vec_env
                    self.target_shape = target_shape
                    self.current_shape = vec_env.observation_space.shape[0]
                    self.pad_amount = target_shape - self.current_shape

                def reset(self):
                    obs = self.vec_env.reset()
                    return self._adapt_obs(obs)

                def step(self, action):
                    obs, reward, done, info = self.vec_env.step(action)
                    obs = self._adapt_obs(obs)
                    return obs, reward, done, info

                def _adapt_obs(self, obs):
                    if self.pad_amount > 0:
                        padding = np.zeros(
                            (obs.shape[0], self.pad_amount), dtype=np.float32
                        )
                        return np.concatenate([obs, padding], axis=1)
                    return obs

                def close(self):
                    self.vec_env.close()

            eval_env = VecObsAdapter(vec_env, model_obs_shape)
        else:
            eval_env = vec_env
    else:
        vec_env.training = False
        vec_env.norm_reward = False
        model = SAC.load(str(model_path), env=model_env)
        print(f"[tracking_eval] ✓ Model loaded with VecNormalize")
        eval_env = vec_env

    print(f"[tracking_eval] Running {n_episodes} episodes on TEST distribution...")

    rewards = []
    tracking_errors = []
    path_progresses = []
    energies = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        ep_tracking = []
        ep_energy = []
        ep_steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_tracking.append(info[0].get("tracking_error", 0.0))
            ep_energy.append(float(np.mean(np.abs(action[0]))))
            ep_steps += 1

            if done[0]:
                path_progresses.append(info[0].get("path_progress", 0.0))

        rewards.append(ep_reward)
        tracking_errors.append(float(np.mean(ep_tracking)))
        energies.append(float(np.mean(ep_energy)))
        episode_lengths.append(ep_steps)

        if (ep + 1) % 10 == 0:
            print(
                f"  {ep + 1}/{n_episodes} | "
                f"tracking_error={np.mean(tracking_errors):.3f}m | "
                f"path_progress={np.mean(path_progresses) * 100:.1f}%"
            )

    eval_env.close()

    # Success = mean tracking error below threshold
    tracking_success = float(np.mean([e < 1.0 for e in tracking_errors]))

    results = {
        "mode": mode,
        "seed": seed,
        "task": "trajectory_tracking",
        "n_episodes": n_episodes,
        "eval_type": "held_out_test",
        "tracking_success_rate": tracking_success,
        "mean_tracking_error": float(np.mean(tracking_errors)),
        "std_tracking_error": float(np.std(tracking_errors)),
        "mean_path_progress": float(np.mean(path_progresses)),
        "std_path_progress": float(np.std(path_progresses)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_energy_per_step": float(np.mean(energies)),
        "std_energy_per_step": float(np.std(energies)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }

    print(f"\n{'=' * 55}")
    print(f"  TRACKING EVAL — {mode.upper()} seed={seed}")
    print(f"  Tracking success:    {results['tracking_success_rate'] * 100:.1f}%")
    print(
        f"  Mean tracking error: {results['mean_tracking_error']:.3f}m "
        f"± {results['std_tracking_error']:.3f}m"
    )
    print(f"  Mean path progress:  {results['mean_path_progress'] * 100:.1f}%")
    print(
        f"  Mean reward:         {results['mean_reward']:.2f} "
        f"± {results['std_reward']:.2f}"
    )
    print(
        f"  Energy/step:         {results['mean_energy_per_step']:.4f} "
        f"± {results['std_energy_per_step']:.4f}"
    )
    print(f"{'=' * 55}")

    save_path = run_dir / "tracking_eval_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[tracking_eval] Saved → {save_path}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--obstacle",
        action="store_true",
        help="Enable ObstacleAUVWrapper (23-dim observation space)",
    )
    args = p.parse_args()
    evaluate_tracking(args.mode, args.seed, args.episodes, use_obstacle=args.obstacle)


if __name__ == "__main__":
    main()
