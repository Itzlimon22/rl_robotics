"""
eval.py — Load a saved model from Drive or local and evaluate it.
Now upgraded for AUV Zero-Shot Transfer (Test Distribution) Evaluation.

Usage:
  python scripts/eval.py --env auv --run-name uniform/uniform_seed0
  python scripts/eval.py --env HalfCheetah-v4 --run-name run_01
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Import custom AUV environment
try:
    from envs.auv_env import HalcyonAUVEnv
except ImportError:
    print("Warning: HalcyonAUVEnv not found. AUV evaluation will fail.")


class TestDRWrapper(gym.Wrapper):
    """
    Forces the environment to use the held-out TEST distribution ranges
    to measure the true Sim-to-Real Gap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # HELD-OUT TEST RANGES (From Research Plan)
        self.drag_range = (0.30, 0.80)
        self.buoyancy_range = (-0.10, 0.15)
        self.current_speed_range = (0.20, 0.60)
        self.added_mass_range = (0.20, 0.50)
        self.efficiency_range = (0.70, 1.00)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Force the test distribution physics
        self.unwrapped.randomize_physics(
            rng=self.unwrapped.np_random,
            drag_range=self.drag_range,
            buoyancy_range=self.buoyancy_range,
            current_speed_range=self.current_speed_range,
            added_mass_range=self.added_mass_range,
            efficiency_range=self.efficiency_range,
        )
        return obs, info


def get_save_dir(env_id, run_name):
    gdrive = "/content/drive/MyDrive/rl_research"
    local = os.path.expanduser("~/rl_research")
    base = gdrive if os.path.exists("/content/drive") else local
    return os.path.join(base, env_id, run_name)


def evaluate(args):
    save_dir = get_save_dir(args.env, args.run_name)
    AlgoCls = SAC if args.algo == "sac" else TD3

    # Load env + normalizer
    if args.env == "auv":

        def make_test_env():
            return TestDRWrapper(HalcyonAUVEnv())

        env = DummyVecEnv([make_test_env])
        print("🌊 AUV Environment Detected: Applying Held-Out Test Distribution")
    else:
        env = make_vec_env(args.env, n_envs=1)

    norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"✅ Loaded observation normalizer from {norm_path}")

    # Load best model (Handling both potential save locations)
    model_path_root = os.path.join(save_dir, "best_model.zip")
    model_path_sub = os.path.join(save_dir, "best_model", "best_model.zip")

    if os.path.exists(model_path_root):
        model_path = model_path_root
    elif os.path.exists(model_path_sub):
        model_path = model_path_sub
    else:
        # Fallback to older format without .zip extension
        model_path = os.path.join(save_dir, "best_model", "best_model")

    model = AlgoCls.load(model_path, env=env)
    print(f"✅ Loaded model from {model_path}\n")

    # Evaluate
    rewards = []
    goal_distances = []
    successes = 0

    print(f"Starting Evaluation ({args.n_episodes} episodes)...")
    for ep in range(args.n_episodes):
        obs = env.reset()
        ep_reward, done = 0, False
        final_dist = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

            if done[0]:
                if args.env == "auv":
                    final_dist = info[0].get("goal_dist", 0.0)
                    if final_dist < 0.5:
                        successes += 1
                    goal_distances.append(final_dist)

        rewards.append(ep_reward)

        if args.env == "auv":
            print(
                f"  Episode {ep + 1:2d}: Reward {ep_reward:8.1f} | Final Dist {final_dist:5.2f}m | {'✅ Success' if final_dist < 0.5 else '❌ Fail'}"
            )
        else:
            print(f"  Episode {ep + 1:2d}: Reward {ep_reward:8.1f}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Mean ± Std Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Min / Max Reward:  {np.min(rewards):.1f} / {np.max(rewards):.1f}")

    if args.env == "auv":
        success_rate = (successes / args.n_episodes) * 100
        print(
            f"Mean Final Dist:   {np.mean(goal_distances):.2f}m ± {np.std(goal_distances):.2f}m"
        )
        print(f"Success Rate:      {success_rate:.1f}% ({successes}/{args.n_episodes})")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="auv")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"])
    p.add_argument("--run-name", default="uniform/uniform_seed0")
    p.add_argument("--n-episodes", type=int, default=100)
    evaluate(p.parse_args())
