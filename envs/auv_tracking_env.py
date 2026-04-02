"""
auv_tracking_env.py — Trajectory tracking task for Halcyon AUV
================================================================================
The AUV must follow a smooth 3D lemniscate (figure-8) path.
The goal marker moves along the path at a fixed speed.
Success = staying within the tracking threshold for the full episode.

This is fundamentally harder than goal-reaching because:
- The target moves continuously.
- The AUV must maintain sustained control authority.
- Physics disturbances cause cumulative path deviation over time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

# Import the base environment
from auv_env import HalcyonAUVEnv


class HalcyonAUVTrackingEnv(HalcyonAUVEnv):
    """
    Trajectory tracking variant of HalcyonAUVEnv.

    Generates a smooth 3D lemniscate path at reset.
    Goal marker follows the path at path_speed m/s.
    Reward encourages staying close to the path.

    Parameters
    ----------
    path_radius : float
        Radius of lemniscate in XY plane (m). Default 4.0.
    path_speed : float
        Speed of path marker (m/s). Default 0.3.
    tracking_threshold : float
        Distance (m) within which tracking is considered good. Default 1.0.
    max_tracking_error : float
        Distance (m) at which episode terminates (too far off path). Default 5.0.
    n_path_points : int
        Resolution of the parametric path. Default 500.
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

        # ANTI-MODE-COLLAPSE FIX:
        # We discovered the base SAC policy prefers to "drift" to save energy
        # when faced with hard currents. For tracking, we must force it to act.
        self.reward_weights["energy"] = 0.005  # Reduced from 0.02
        self.reward_weights["progress"] = 15.0  # Increased from 10.0

        # Path state variables
        self._path_points: np.ndarray = np.zeros((n_path_points, 3))
        self._path_idx: int = 0
        self._path_t: float = 0.0
        self._tracking_errors: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Resets the AUV and generates a new 3D path."""
        # Reset base environment (places AUV at origin, handles DR)
        obs, info = super().reset(seed=seed, options=options)

        # Generate lemniscate path
        self._path_points = self._generate_lemniscate()
        self._path_idx = 0
        self._path_t = 0.0
        self._tracking_errors = []

        # Override the random goal placement from super().reset()
        # Place goal at the very first path point
        self._goal_pos = self._path_points[0].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._goal_distance()

        obs = self._get_observation()
        info["tracking_error"] = 0.0
        info["path_progress"] = 0.0
        info["is_success"] = False
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advances the simulation and moves the tracking marker."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Advance path marker at constant speed
        self._advance_path()

        # Compute tracking error (distance from AUV to current path point)
        auv_pos = np.array(info["auv_pos"])
        tracking_error = float(
            np.linalg.norm(auv_pos - self._path_points[self._path_idx])
        )
        self._tracking_errors.append(tracking_error)

        # Apply Tracking-specific Rewards
        tracking_reward = -0.5 * tracking_error  # penalise deviation from path
        if tracking_error < self.tracking_threshold:
            tracking_reward += 2.0  # strong bonus for staying tight on the path

        reward += tracking_reward

        # Terminate early if the AUV gets completely blown off course
        if tracking_error > self.max_tracking_error:
            terminated = True

        path_progress = self._path_idx / max(self.n_path_points - 1, 1)

        # If we survive the whole episode without violating max_tracking_error,
        # it is considered a successful tracking run.
        is_success = bool(truncated and tracking_error < self.tracking_threshold * 2)

        info["tracking_error"] = tracking_error
        info["path_progress"] = path_progress
        info["mean_tracking_error"] = float(np.mean(self._tracking_errors))
        info["is_success"] = (
            is_success  # Consumed by AUVDomainRandomWrapper for curriculum updates
        )

        return obs, reward, terminated, truncated, info

    def _generate_lemniscate(self) -> np.ndarray:
        """
        Generate a smooth 3D lemniscate (figure-8) path.
        Parametric equations:
            x(t) = R * cos(t) / (1 + sin^2(t))
            y(t) = R * sin(t)*cos(t) / (1 + sin^2(t))
            z(t) = A * sin(2t)   (gentle vertical oscillation)
        """
        R = self.path_radius
        A = 1.5  # vertical amplitude (m)
        t = np.linspace(0, 2 * np.pi, self.n_path_points)

        denom = 1 + np.sin(t) ** 2
        x = R * np.cos(t) / denom
        y = R * np.sin(t) * np.cos(t) / denom
        z = A * np.sin(2 * t)

        # Clamp z to valid depth range to prevent hitting seabed/surface
        z = np.clip(z, -6.0, 6.0)

        return np.column_stack([x, y, z])

    def _advance_path(self):
        """Move the physical goal marker along the path based on path_speed."""
        dt = self.effective_dt
        step_distance = self.path_speed * dt

        # Move to next path point if we've consumed the step_distance
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

        # Wrap path (loop forever if episode is long)
        if self._path_idx >= self.n_path_points - 1:
            self._path_idx = 0

        # Update goal marker in MuJoCo
        self._goal_pos = self._path_points[self._path_idx].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

    def get_path_points(self) -> np.ndarray:
        """Return full path for downstream rendering/visualisation."""
        return self._path_points.copy()


if __name__ == "__main__":
    print("Testing HalcyonAUVTrackingEnv...")
    env = HalcyonAUVTrackingEnv(path_speed=0.5)
    obs, info = env.reset()
    for _ in range(50):
        obs, r, t, tr, info = env.step(env.action_space.sample())

    print(f"Path Progress after 50 steps: {info['path_progress'] * 100:.1f}%")
    print(f"Tracking Error: {info['tracking_error']:.2f}m")
    env.close()
    print("Tracking Env OK!")
