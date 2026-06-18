#!/usr/bin/env python3
"""
viewer_interactive.py — Interactive MuJoCo Viewer for AUV
=========================================================

Open the AUV environment in MuJoCo's interactive viewer.
Use mouse/keyboard to adjust camera angle and zoom for perfect paper figure.

Usage:
    conda activate rl
    cd ~/rl_robotics
    python scripts/viewer_interactive.py

Controls:
    LEFT MOUSE + DRAG    — Rotate view
    RIGHT MOUSE + DRAG   — Zoom in/out
    SCROLL               — Zoom
    SPACE                — Play/pause
    R                    — Reset to home pose
    P                    — Print current camera position
    ESC                  — Exit

"""

import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# ── Path setup ────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper


def main():
    print("\n" + "=" * 70)
    print("MuJoCo Interactive Viewer — AUV Environment")
    print("=" * 70)
    print("\nControls:")
    print("  LEFT MOUSE + DRAG   — Rotate camera")
    print("  RIGHT MOUSE + DRAG  — Zoom")
    print("  SCROLL              — Zoom in/out")
    print("  SPACE               — Play/pause simulation")
    print("  R                   — Reset to home")
    print("  P                   — Print camera info")
    print("  ESC                 — Exit")
    print("\nTip: Find the perfect camera angle, then note the position in the viewer!")
    print("=" * 70 + "\n")

    # Create environment
    xml_path = _REPO_ROOT / "envs" / "auv.xml"
    base_env = HalcyonAUVEnv(xml_path=str(xml_path))
    env = AUVDomainRandomWrapper(base_env, mode="none", seed=42, verbose=False)

    # Reset to get initial state
    obs, info = env.reset()
    print(f"✓ Environment loaded: {xml_path}")
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ AUV at: {info['auv_pos']}")
    print(f"✓ Goal at: {info['goal_pos']}\n")

    # Open MuJoCo viewer (interactive)
    print("Opening MuJoCo viewer... (this may take 5-10 seconds)")
    print("Once loaded, use mouse to rotate camera for perfect view.\n")

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        # Hide physics overlays
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        # Set initial camera position - START WIDER SO YOU CAN SEE THE AUV
        viewer.cam.distance = 8.0  # Wide view initially
        viewer.cam.elevation = -35.0  # Angled down
        viewer.cam.azimuth = 120.0  # Good view
        viewer.cam.lookat[:] = [0, 0, 0.0]  # Look at origin

        print("\n" + "=" * 70)
        print(
            "Viewer is LIVE - You should see the AUV in center (white background in export)!"
        )
        print("=" * 70)
        print("\n✓ If AUV not visible: Try rotating with LEFT MOUSE")
        print("✓ RIGHT MOUSE + DRAG DOWN to zoom in close")
        print("✓ Camera values print every 2 seconds below...\n")

        step_counter = 0
        last_print = 0
        while viewer.is_running():
            step_counter += 1

            # Every 2 seconds, show current camera position on screen
            if step_counter - last_print > 50:  # ~2 seconds at ~25fps
                dist = viewer.cam.distance
                elev = viewer.cam.elevation
                azim = viewer.cam.azimuth
                print(
                    f"Current camera: distance={dist:.2f}, elevation={elev:.1f}°, azimuth={azim:.1f}°"
                )
                last_print = step_counter

                # Also show in viewer title
                viewer.sync()

            # Keep AUV stationary so we can take a photo
            # action = base_env.action_space.sample()
            # obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            if step_counter == 1:
                print("💡 Camera distance/elevation/azimuth shown above periodically")
                print("💡 When you find perfect angle, copy those values\n")


if __name__ == "__main__":
    main()
