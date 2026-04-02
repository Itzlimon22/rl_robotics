"""
auv_moving_goal.py — Moving goal wrapper for HalcyonAUVEnv
================================================================================
Wraps HalcyonAUVEnv so the goal drifts slowly during each episode.
The goal moves at a constant velocity in a random direction,
bouncing off workspace boundaries.

This tests whether policies trained with CDR generalise to
dynamic goal-following, not just static goal-reaching.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import Wrapper


class MovingGoalWrapper(Wrapper):
    """
    Wrapper that makes the goal drift at constant speed during each episode.

    The goal starts at a random position (same as base env) and moves
    at goal_speed m/s in a random direction. When it hits the workspace
    boundary, it reflects (bounces) to stay within the valid volume.

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment.
    goal_speed : float
        Speed of goal movement in m/s. Default 0.3.
        0.0 = static goal (same as base env).
        0.3 = slow drift (realistic ocean current speed).
        0.8 = fast moving target.
    workspace_radius : float
        Boundary for goal reflection. Should match env.workspace_radius.
    """

    def __init__(
        self,
        env,
        goal_speed: float = 0.3,
        workspace_radius: float = 12.0,
    ):
        super().__init__(env)
        self.goal_speed = goal_speed
        self.workspace_radius = workspace_radius
        self._goal_velocity = np.zeros(3, dtype=np.float64)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Resets the environment and calculates a new random drift vector."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Generate a random 3D direction vector
        direction = self.env.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8

        # Keep goal mostly horizontal (real AUV targets rarely shoot straight up/down)
        direction[2] *= 0.3
        direction /= np.linalg.norm(direction) + 1e-8

        self._goal_velocity = direction * self.goal_speed
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advances the simulation and moves the goal position continuously."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Move the goal marker in the physics engine
        self._update_goal_position()

        # Recompute the observation vector because the relative goal distance/angle changed
        obs = self.env._get_observation()

        # Inject the new goal metrics into the info dictionary for logging
        info["goal_pos"] = self.env._goal_pos.tolist()
        info["goal_speed"] = float(self.goal_speed)

        return obs, reward, terminated, truncated, info

    def _update_goal_position(self):
        """Moves goal by one control step and calculates boundary reflections."""
        dt = self.env.effective_dt
        new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Reflect off the outer workspace sphere
        dist = np.linalg.norm(new_pos)
        if dist > self.workspace_radius * 0.8:
            # Calculate normal vector and reflect velocity
            normal = new_pos / (dist + 1e-8)
            self._goal_velocity -= 2 * np.dot(self._goal_velocity, normal) * normal
            new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Clamp Z to ocean volume so the goal doesn't fly out of the water or hit the seabed
        new_pos[2] = np.clip(new_pos[2], -8.0, 8.0)

        # Update goal position in the MuJoCo engine
        self.env._goal_pos = new_pos
        self.env._set_goal_body(new_pos)
        mujoco.mj_forward(self.env.model, self.env.data)


def make_moving_goal_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    goal_speed: float = 0.3,
):
    """
    Factory function to build the fully stacked Moving Goal environment.
    Stack: HalcyonAUVEnv -> MovingGoalWrapper -> AUVDomainRandomWrapper
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from auv_env import HalcyonAUVEnv
    from auv_dr_wrapper import AUVDomainRandomWrapper

    base = HalcyonAUVEnv(xml_path=xml_path)
    moving = MovingGoalWrapper(base, goal_speed=goal_speed)
    return AUVDomainRandomWrapper(moving, mode=mode, seed=seed)


# ==============================================================================
# Quick Verification Script
# ==============================================================================
if __name__ == "__main__":
    from auv_env import HalcyonAUVEnv

    print("Testing MovingGoalWrapper...")
    env = MovingGoalWrapper(HalcyonAUVEnv(), goal_speed=0.3)
    obs, info = env.reset(seed=0)
    pos0 = np.array(info["goal_pos"])

    for _ in range(10):
        obs, r, t, tr, info = env.step(env.action_space.sample())

    pos1 = np.array(info["goal_pos"])

    print(f"Start Pos: {pos0}")
    print(f"End Pos:   {pos1}")
    print("Goal moved correctly:", not np.allclose(pos0, pos1))
    print(f"Distance moved: {np.linalg.norm(pos1 - pos0):.4f} m")
    env.close()
