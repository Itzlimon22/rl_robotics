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
from gymnasium.spaces import Box

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
    lookahead_time : float
        Time (seconds) to look ahead for the Pure Pursuit vector. Default 2.0.
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
        lookahead_time: float = 2.0,
        tracking_threshold: float = 1.0,
        max_tracking_error: float = 5.0,
        n_path_points: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_radius = path_radius
        self.path_speed = path_speed
        self.lookahead_time = lookahead_time
        self.tracking_threshold = tracking_threshold
        self.max_tracking_error = max_tracking_error
        self.n_path_points = n_path_points

        # ANTI-MODE-COLLAPSE FIX:
        # Force the agent to act instead of drifting to save energy.
        self.reward_weights["energy"] = 0.005
        self.reward_weights["progress"] = 15.0

        # Path state variables
        self._path_points: np.ndarray = np.zeros((n_path_points, 3))
        self._path_idx: int = 0
        self._path_t: float = 0.0
        self._tracking_errors: list = []

        # PROFESSIONAL POLISH: Dynamically extend observation space for Pure Pursuit
        # This prevents shape mismatch errors when the Obstacle Wrapper is applied later
        old_shape = self.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(old_shape + 3,), dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the AUV and generates a new 3D parametric path.
        """
        # Generate lemniscate path first to prevent zero-distance loops when super().reset() calls _get_observation
        self._path_points = self._generate_lemniscate()
        self._path_idx = 0
        self._path_t = 0.0
        self._tracking_errors = []

        # Reset base environment (places AUV at origin, handles DR)
        obs, info = super().reset(seed=seed, options=options)

        # Override the random goal placement from super().reset()
        self._goal_pos = self._path_points[0].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._goal_distance()

        # Re-fetch observation to include the new lookahead vector
        obs = self._get_observation()

        info["tracking_error"] = 0.0
        info["path_progress"] = 0.0
        info["is_success"] = False
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advances the simulation, moves the tracking marker, and computes path rewards.
        """
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
        tracking_reward = -0.5 * tracking_error
        if tracking_error < self.tracking_threshold:
            tracking_reward += 2.0  # Strong bonus for staying tight on the path

        reward += tracking_reward

        # Early Return / Termination trigger
        if tracking_error > self.max_tracking_error:
            terminated = True

        path_progress = self._path_idx / max(self.n_path_points - 1, 1)
        is_success = bool(truncated and tracking_error < self.tracking_threshold * 2)

        info["tracking_error"] = tracking_error
        info["path_progress"] = path_progress
        info["mean_tracking_error"] = float(np.mean(self._tracking_errors))
        info["is_success"] = is_success

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Appends the 3D Pure Pursuit lookahead vector to the base observations.
        """
        # 1. Fetch base observations (19-dim base or 23-dim obstacle)
        base_obs = super()._get_observation()

        # 2. Calculate future target position
        lookahead_distance = self.path_speed * self.lookahead_time
        accumulated_dist = 0.0
        future_idx = self._path_idx

        # Traverse path until we reach the lookahead distance
        max_lookahead_steps = self.n_path_points
        steps = 0
        while accumulated_dist < lookahead_distance and steps < max_lookahead_steps:
            next_idx = (future_idx + 1) % self.n_path_points
            segment_len = np.linalg.norm(
                self._path_points[next_idx] - self._path_points[future_idx]
            )
            accumulated_dist += segment_len
            future_idx = next_idx
            steps += 1

        future_goal_pos_world = self._path_points[future_idx]

        # 3. Transform future global vector into AUV's local body frame
        auv_pos_world = self.data.qpos[:3]
        global_lookahead_vec = future_goal_pos_world - auv_pos_world

        # Use MuJoCo's rotation matrix (xmat) for the AUV body (Body ID 1 is standard)
        # Transposing the world rotation matrix gives the local rotation matrix
        auv_rotation_matrix = self.data.xmat[1].reshape(3, 3)
        local_lookahead_vec = auv_rotation_matrix.T @ global_lookahead_vec

        # 4. Concatenate and return
        return np.concatenate([base_obs, local_lookahead_vec], dtype=np.float32)

    def _generate_lemniscate(self) -> np.ndarray:
        """
        Generate a smooth 3D lemniscate (figure-8) path.
        """
        R = self.path_radius
        A = 1.5
        t = np.linspace(0, 2 * np.pi, self.n_path_points)

        denom = 1 + np.sin(t) ** 2
        x = R * np.cos(t) / denom
        y = R * np.sin(t) * np.cos(t) / denom
        z = A * np.sin(2 * t)

        z = np.clip(z, -6.0, 6.0)

        return np.column_stack([x, y, z])

    def _advance_path(self) -> None:
        """
        Moves the physical goal marker forward along the path.
        """
        dt = self.effective_dt
        step_distance = self.path_speed * dt

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

        if self._path_idx >= self.n_path_points - 1:
            self._path_idx = 0

        self._goal_pos = self._path_points[self._path_idx].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

    def get_path_points(self) -> np.ndarray:
        """Return full path for downstream rendering/visualisation."""
        return self._path_points.copy()


if __name__ == "__main__":
    print("Testing HalcyonAUVTrackingEnv with Lookahead...")
    env = HalcyonAUVTrackingEnv(path_speed=0.5)
    obs, info = env.reset()
    for _ in range(50):
        obs, r, t, tr, info = env.step(env.action_space.sample())

    print(f"Observation Shape: {obs.shape}")
    print(f"Path Progress after 50 steps: {info['path_progress'] * 100:.1f}%")
    print(f"Tracking Error: {info['tracking_error']:.2f}m")
    env.close()
    print("Tracking Env OK!")
