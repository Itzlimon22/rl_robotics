"""
auv_tracking_env.py — Trajectory tracking task for Halcyon AUV
===============================================================
The AUV must follow a smooth 3D lemniscate (figure-8) path.
The goal marker moves along the path at a fixed speed.
Success = staying within 1.0m of the path for the full episode.
"""

from __future__ import annotations
import numpy as np
import mujoco
from typing import Optional, Dict, Any, Tuple

# Assuming this is accessible based on your structure
from envs.auv_env import HalcyonAUVEnv


class HalcyonAUVTrackingEnv(HalcyonAUVEnv):
    """
    Trajectory tracking variant of HalcyonAUVEnv.
    Generates a smooth 3D lemniscate path at reset.
    Goal marker follows the path at path_speed m/s.
    """

    def __init__(
        self,
        path_radius: float = 4.0,
        path_speed: float = 0.3,
        tracking_threshold: float = 1.0,
        max_tracking_error: float = 5.0,
        n_path_points: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_radius = path_radius
        self.path_speed = path_speed
        self.tracking_threshold = tracking_threshold
        self.max_tracking_error = max_tracking_error
        self.n_path_points = n_path_points

        # Path state
        self._path_points: np.ndarray = np.zeros((n_path_points, 3))
        self._path_idx: int = 0
        self._path_t: float = 0.0
        self._tracking_errors: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # Reset base environment
        obs, info = super().reset(seed=seed, options=options)

        # Generate lemniscate path
        self._path_points = self._generate_lemniscate()
        self._path_idx = 0
        self._path_t = 0.0
        self._tracking_errors = []

        # Place goal at first path point
        self._goal_pos = self._path_points[0].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._goal_distance()

        obs = self._get_observation()
        info["tracking_error"] = 0.0
        info["path_progress"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Advance path marker
        self._advance_path()

        # Compute tracking error
        auv_pos = np.array(info["auv_pos"])
        tracking_error = float(
            np.linalg.norm(auv_pos - self._path_points[self._path_idx])
        )
        self._tracking_errors.append(tracking_error)

        # Add tracking reward component
        tracking_reward = -0.5 * tracking_error  # penalise deviation from path
        if tracking_error < self.tracking_threshold:
            tracking_reward += 2.0  # bonus for staying on path

        reward += tracking_reward

        # Terminate if too far off path
        if tracking_error > self.max_tracking_error:
            terminated = True

        path_progress = self._path_idx / max(self.n_path_points - 1, 1)

        info["tracking_error"] = tracking_error
        info["path_progress"] = path_progress
        info["mean_tracking_error"] = float(np.mean(self._tracking_errors))

        return obs, reward, terminated, truncated, info

    def _generate_lemniscate(self) -> np.ndarray:
        """
        Generate a smooth 3D lemniscate (figure-8) path.
        """
        R = self.path_radius
        A = 1.5  # vertical amplitude (m)
        t = np.linspace(0, 2 * np.pi, self.n_path_points)

        denom = 1 + np.sin(t) ** 2
        x = R * np.cos(t) / denom
        y = R * np.sin(t) * np.cos(t) / denom
        z = A * np.sin(2 * t)

        # Clamp z to valid depth range
        z = np.clip(z, -6.0, 6.0)

        return np.column_stack([x, y, z])

    def _advance_path(self):
        """Move goal marker along path based on path_speed."""
        dt = self.effective_dt
        step_distance = self.path_speed * dt

        # Move to next path point if close enough
        while self._path_idx < self.n_path_points - 1:
            next_idx = self._path_idx + 1
            segment_len = np.linalg.norm(
                self._path_points[next_idx] - self._path_points[self._path_idx]
            )
            if step_distance >= segment_len:
                step_distance -= segment_len
                self._path_idx = next_idx
            else:
                break

        # Wrap path (loop forever)
        if self._path_idx >= self.n_path_points - 1:
            self._path_idx = 0

        # Update goal marker
        self._goal_pos = self._path_points[self._path_idx].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

    def get_path_points(self) -> np.ndarray:
        """Return full path for visualisation."""
        return self._path_points.copy()
