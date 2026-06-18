#!/usr/bin/env python3
"""
Simple MuJoCo Viewer - See everything without fiddling with cameras
"""

import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from auv_env import HalcyonAUVEnv


def main():
    print("\n" + "=" * 70)
    print("Simple MuJoCo Viewer - TOP-DOWN VIEW")
    print("=" * 70)

    # Create environment (NO wrapper, just raw env)
    xml_path = _REPO_ROOT / "envs" / "auv.xml"
    env = HalcyonAUVEnv(xml_path=str(xml_path))

    # Reset to initialize
    obs, info = env.reset()
    print(f"\n✓ AUV position: {info['auv_pos']}")
    print(f"✓ Goal position: {info['goal_pos']}")

    print("\nOpening viewer in TOP-DOWN mode...")
    print("You should see:")
    print("  - Yellow AUV (center)")
    print("  - Green goal sphere")
    print("  - Blue arena floor\n")

    # Open viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Disable rendering of joints/actuators
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

        # Start with TOP-DOWN view
        viewer.cam.distance = 15.0
        viewer.cam.elevation = -89.0  # Almost straight down
        viewer.cam.azimuth = 0.0
        viewer.cam.lookat[:] = [0, 0, 0]

        viewer.sync()

        print("=" * 70)
        print("Controls:")
        print("  LEFT MOUSE + DRAG   → Rotate")
        print("  RIGHT MOUSE + DRAG  → Zoom")
        print("  SCROLL              → Zoom")
        print("\nIf you still don't see the AUV:")
        print("  1. Try rotating with LEFT MOUSE")
        print("  2. Use SCROLL to zoom in/out")
        print("=" * 70 + "\n")

        step = 0
        while viewer.is_running():
            step += 1

            # Every 2 seconds, print camera values
            if step % 50 == 0:
                d = viewer.cam.distance
                e = viewer.cam.elevation
                a = viewer.cam.azimuth
                print(
                    f"[{step // 50 * 2}s] distance={d:.1f}, elevation={e:.1f}°, azimuth={a:.1f}° | Found AUV? (Ctrl+C when happy)"
                )

            # Random action
            action = env.action_space.sample()
            env.step(action)
            viewer.sync()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nViewer closed. Use those camera values for render_video.py!\n")
