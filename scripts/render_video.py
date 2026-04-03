"""
render_video.py — Hard-Locked Tracking Fix
===================================================================
Manually synchronizes the camera look-at point with the AUV's global
position to prevent it from ever leaving the screen.
"""

import argparse
import sys
import gymnasium as gym
import mujoco
import imageio
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Standard Path setup
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

try:
    from auv_env import HalcyonAUVEnv
    from auv_dr_wrapper import AUVDomainRandomWrapper
except ImportError:
    from envs.auv_env import HalcyonAUVEnv
    from envs.auv_dr_wrapper import AUVDomainRandomWrapper


class RangefinderObsWrapper(gym.Wrapper):
    """Bridges 19-dim env to 23-dim model observations."""

    def __init__(self, env):
        super().__init__(env)
        low = np.append(env.observation_space.low, np.full(4, -1.0, dtype=np.float32))
        high = np.append(env.observation_space.high, np.full(4, 20.0, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self, obs):
        try:
            rf = self.unwrapped.data.sensordata[23:27].copy().astype(np.float32)
        except:
            rf = np.full(4, -1.0, dtype=np.float32)
        return np.append(obs, rf).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._get_obs(obs), reward, term, trunc, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Find the model directory
    base = Path.home() / "rl_research" / "auv"
    run_dir = next(
        (
            p
            for p in [
                base / f"master_{args.mode}" / f"master_{args.mode}_seed{args.seed}",
                base / args.mode / f"{args.mode}_seed{args.seed}",
            ]
            if p.exists()
        ),
        None,
    )

    if not run_dir:
        print(
            f"❌ Error: Folder not found for {args.mode}. Check your directory structure."
        )
        return

    model_path, vn_path = run_dir / "best_model.zip", run_dir / "vec_normalize.pkl"
    xml_path = str(_REPO_ROOT / "envs" / "auv.xml")

    # Env Setup
    raw_env = HalcyonAUVEnv(xml_path=xml_path, render_mode="rgb_array")
    env = AUVDomainRandomWrapper(raw_env, mode="none", seed=args.seed)

    # Check Model Compatibility
    model = SAC.load(model_path)
    if model.observation_space.shape[0] == 23:
        env = RangefinderObsWrapper(env)

    # Wrap for Vector Normalization
    venv = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    model.set_env(venv)

    # --- Pre-calculate Body ID for Speed ---
    core = env.unwrapped
    auv_body_id = -1
    for i in range(core.model.nbody):
        name = mujoco.mj_id2name(core.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(x in name.lower() for x in ["auv", "base", "robot"]):
            auv_body_id = i
            break

    if auv_body_id == -1:
        auv_body_id = 1  # Fallback to first body

    frames = []
    obs = venv.reset()
    print(f"🎬 Rendering {args.mode.upper()} with Hard-Locked Camera...")

    for i in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)

        # --- HARD-LOCKED CAMERA SYNC ---
        # Get absolute world position of the AUV body
        auv_pos = core.data.xpos[auv_body_id].copy()

        if hasattr(core, "mujoco_renderer"):
            viewer = core.mujoco_renderer._get_viewer(render_mode="rgb_array")
            if viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                # Center the camera on the AUV
                viewer.cam.lookat[:] = auv_pos
                # Tune these numbers: 15.0 is close, 30.0 is far
                viewer.cam.distance = 20.0
                viewer.cam.elevation = -90.0  # Straight top-down
                viewer.cam.azimuth = 90.0

        # Capture the frame
        frames.append(env.render())

        if done[0]:
            break
        if i % 100 == 0:
            print(f"   Progress: {i}/800 frames...")

    # Save to Paper Assets
    out_dir = Path.home() / "rl_research" / "paper_assets" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"locked_tracking_{args.mode}.mp4"
    imageio.mimwrite(out_file, frames, fps=25, quality=9)
    print(f"\n✅ SUCCESS! Video saved to: {out_file}")


if __name__ == "__main__":
    main()
