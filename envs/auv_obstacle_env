"""
auv_obstacle_env.py — Obstacle avoidance wrapper for HalcyonAUVEnv
===================================================================
Adds spherical obstacles placed randomly each episode.
Extends observation with rangefinder readings.
Adds collision penalty to reward.
"""

from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import Wrapper, spaces
from typing import Optional, Dict, Any, Tuple

from envs.auv_env import HalcyonAUVEnv


class ObstacleAUVWrapper(Wrapper):
    """
    Adds obstacle avoidance to HalcyonAUVEnv.
    Changes:
    - Observation expanded: +4 rangefinder readings (fwd, port, stbd, up)
    - Reward: -collision_penalty when AUV hits an obstacle
    - Reset: randomises obstacle positions each episode
    """

    N_RANGEFINDERS = 4  # forward, port, starboard, up

    def __init__(
        self,
        env: HalcyonAUVEnv,
        n_obstacles: int = 5,
        obstacle_radius: float = 0.4,
        collision_penalty: float = 10.0,
        min_obstacle_dist: float = 1.5,
    ):
        super().__init__(env)
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.collision_penalty = collision_penalty
        self.min_obstacle_dist = min_obstacle_dist

        # Extend observation space with rangefinder readings
        base_obs = self.env.observation_space
        low = np.append(base_obs.low, np.full(self.N_RANGEFINDERS, -1.0))
        high = np.append(base_obs.high, np.full(self.N_RANGEFINDERS, 10.0))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Cache obstacle geom IDs
        self._obs_geom_ids = []
        for i in range(n_obstacles):
            try:
                gid = mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}"
                )
                self._obs_geom_ids.append(gid)
            except Exception:
                pass  # geom not in XML — skip

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Randomise obstacle positions
        self._randomise_obstacles()

        # Get extended observation
        obs = self._get_extended_obs(obs)
        info["collision"] = False
        info["min_obstacle_dist"] = self._min_obstacle_dist()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check collision
        collision = self._check_collision()
        if collision:
            reward -= self.collision_penalty
            terminated = True  # End episode immediately on crash

        obs = self._get_extended_obs(obs)
        info["collision"] = collision
        info["min_obstacle_dist"] = self._min_obstacle_dist()

        # Update success metric: Only successful if it reached goal AND didn't crash
        if terminated and collision:
            info["is_success"] = False

        return obs, reward, terminated, truncated, info

    def _randomise_obstacles(self):
        """Place obstacles at random positions, avoiding AUV start and goal."""
        auv_pos = np.zeros(3)  # AUV starts at origin
        goal_pos = self.env._goal_pos

        placed = 0
        attempts = 0

        for i, gid in enumerate(self._obs_geom_ids):
            while attempts < 100:
                attempts += 1
                # Random position within workspace
                r = self.env.np_random.uniform(1.5, 8.0)
                ang = self.env.np_random.uniform(0, 2 * np.pi)
                elv = self.env.np_random.uniform(-np.pi / 4, np.pi / 4)
                pos = np.array(
                    [
                        r * np.cos(elv) * np.cos(ang),
                        r * np.cos(elv) * np.sin(ang),
                        r * np.sin(elv),
                    ]
                )
                pos[2] = np.clip(pos[2], -6.0, 6.0)

                # Check clearance from AUV start position and Goal position
                if (
                    np.linalg.norm(pos - auv_pos) > self.min_obstacle_dist
                    and np.linalg.norm(pos - goal_pos) > self.min_obstacle_dist
                ):
                    self.env.model.geom_pos[gid] = pos
                    placed += 1
                    break

        mujoco.mj_forward(self.env.model, self.env.data)

    def _check_collision(self) -> bool:
        """True if any contact involves the AUV body and an obstacle geom."""
        for i in range(self.env.data.ncon):
            con = self.env.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 in self._obs_geom_ids or g2 in self._obs_geom_ids:
                return True
        return False

    def _min_obstacle_dist(self) -> float:
        """Minimum distance from AUV to any obstacle."""
        auv_pos = self.env.data.qpos[:3]
        min_dist = float("inf")
        for gid in self._obs_geom_ids:
            obs_pos = self.env.model.geom_pos[gid]
            d = float(np.linalg.norm(auv_pos - obs_pos)) - self.obstacle_radius
            min_dist = min(min_dist, d)
        return max(min_dist, 0.0)

    def _get_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append 4 rangefinder readings to base observation."""
        # Assuming rangefinders are indices 23-26 in sensordata based on Halcyon config
        # If your XML doesn't have sensors yet, we will pad with zeros for now
        try:
            rf = self.env.data.sensordata[23:27].copy()
        except IndexError:
            # Fallback if sensors aren't defined in XML yet
            rf = np.full(self.N_RANGEFINDERS, 10.0)

        return np.append(base_obs, rf.astype(np.float32))
