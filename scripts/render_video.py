"""
render_video.py — Phase 7: Professional Tracking Video Generation
===================================================================
Renders a trained SAC policy in the Obstacle environment.
Uses direct MuJoCo camera access to ensure the AUV is tracked.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import imageio
import mujoco

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [_REPO_ROOT, _REPO_ROOT / "envs"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.auv_dr_wrapper import make_obstacle_env


def generate_video(model_dir: str, mode: str, seed: int, n_episodes: int = 3):
    model_path = Path(model_dir) / "final_model.zip"
    vec_path = Path(model_dir) / "vec_normalize.pkl"
    xml_path = _REPO_ROOT / "envs" / "auv.xml"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Create Env
    def _init():
        # Force render_mode to rgb_array for frame capture
        env = make_obstacle_env(str(xml_path), mode=mode, seed=seed)
        env.unwrapped.render_mode = "rgb_array"
        env.unwrapped.width = 1280
        env.unwrapped.height = 720
        env.set_test_distribution()
        return env

    venv = DummyVecEnv([_init])
    env = VecNormalize.load(str(vec_path), venv)
    env.training = False
    env.norm_reward = False

    model = SAC.load(str(model_path), env=env)

    frames = []
    print(f"Recording {n_episodes} episodes for {mode}...")

    # Access the base env
    unwrapped = env.venv.envs[0].unwrapped

    # Try to find the renderer attribute (Gymnasium version fallback)
    renderer = None
    for attr in ["_renderer", "renderer", "mujoco_renderer"]:
        if hasattr(unwrapped, attr):
            renderer = getattr(unwrapped, attr)
            break

    if renderer is None:
        # If no renderer attr, we force initialize it via the first render call
        _ = unwrapped.render()
        renderer = getattr(unwrapped, "renderer", getattr(unwrapped, "_renderer", None))

    for i in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            # --- THE CAMERA TRACKING LOGIC ---
            # Define a camera that tracks the AUV body
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = unwrapped._auv_body_id
            cam.distance = 7.0
            cam.elevation = -25

            # Capture frame using the specific camera configuration
            # In new Gymnasium, we often need to use the viewer or the renderer directly
            try:
                # Attempt to render with the tracking camera
                frame = unwrapped.render()
                # If the frame is empty or the camera is wrong, we manually
                # update the scene with our tracking cam
                if hasattr(renderer, "update_scene"):
                    renderer.update_scene(unwrapped.data, camera=cam)
                    frame = renderer.render()
            except Exception:
                # Absolute fallback: just render whatever the env gives us
                frame = unwrapped.render()

            frames.append(frame)
            step += 1

        print(f"  Episode {i + 1} complete ({step} steps)")

    env.close()

    # 2. Save Video
    output_name = f"auv_final_tracking_{mode}.mp4"
    imageio.mimwrite(output_name, frames, fps=25, quality=9)
    print(f"\nSuccess! Video saved as: {output_name}")
    print(f"Location: {Path.cwd() / output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["none", "uniform", "curriculum"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=int, default=3)
    args = parser.parse_args()

    generate_video(args.model_dir, args.mode, args.seed, args.eps)
