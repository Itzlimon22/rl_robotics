"""
auv_moving_goal.py — Moving goal wrapper for HalcyonAUVEnv
===========================================================
Wraps HalcyonAUVEnv so the goal drifts slowly during each episode.
The goal moves at a constant velocity in a random direction,
bouncing off workspace boundaries.

This tests whether policies trained with CDR generalise to
dynamic goal-following, not just static goal-reaching.
"""

from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import Wrapper
from typing import Optional, Dict, Any, Tuple


class MovingGoalWrapper(Wrapper):
    """
    Wrapper that makes the goal drift at constant speed during each episode.

    The goal starts at a random position (same as base env) and moves
    at goal_speed m/s in a random direction. When it hits the workspace
    boundary, it reflects (bounces).
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
        obs, info = self.env.reset(seed=seed, options=options)

        # Random goal velocity direction
        direction = self.env.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8

        # Keep goal mostly horizontal (real AUV targets don't move vertically fast)
        direction[2] *= 0.3
        direction /= np.linalg.norm(direction) + 1e-8

        self._goal_velocity = direction * self.goal_speed
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Move goal
        self._update_goal_position()

        # Recompute observation with new goal position
        obs = self.env._get_observation()
        info["goal_pos"] = self.env._goal_pos.tolist()
        info["goal_speed"] = float(self.goal_speed)

        return obs, reward, terminated, truncated, info

    def _update_goal_position(self):
        """Move goal by one control step. Reflect off boundaries."""
        dt = self.env.effective_dt
        new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Reflect off workspace sphere
        dist = np.linalg.norm(new_pos)
        if dist > self.workspace_radius * 0.8:
            # Reflect velocity
            normal = new_pos / (dist + 1e-8)
            self._goal_velocity -= 2 * np.dot(self._goal_velocity, normal) * normal
            new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Clamp Z to ocean volume
        new_pos[2] = np.clip(new_pos[2], -8.0, 8.0)

        # Update goal in MuJoCo
        self.env._goal_pos = new_pos
        self.env._set_goal_body(new_pos)
        mujoco.mj_forward(self.env.model, self.env.data)


# --- Factory Function ---
def make_moving_goal_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    goal_speed: float = 0.3,
):
    """
    Create a moving goal env with DR wrapper.
    Stack: HalcyonAUVEnv → MovingGoalWrapper → AUVDomainRandomWrapper
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from auv_env import HalcyonAUVEnv
    from auv_dr_wrapper import AUVDomainRandomWrapper

    base = HalcyonAUVEnv(xml_path=xml_path)
    moving = MovingGoalWrapper(base, goal_speed=goal_speed)
    return AUVDomainRandomWrapper(moving, mode=mode, seed=seed)
