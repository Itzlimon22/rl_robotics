"""
auv_obstacle_env.py — Obstacle avoidance wrapper for HalcyonAUVEnv
================================================================================
Adds N_OBSTACLES spherical obstacles placed randomly each episode.
Extends observation with 4 rangefinder readings (simulated sonar).
Adds collision penalty to the reward function and terminates on crash.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import Wrapper, spaces

from auv_env import HalcyonAUVEnv


class ObstacleAUVWrapper(Wrapper):
    """
    Adds obstacle avoidance to HalcyonAUVEnv.

    Changes:
    - Observation expanded: +4 rangefinder readings (fwd, port, stbd, up)
    - Reward: -collision_penalty when AUV hits an obstacle
    - Reset: randomises obstacle positions each episode safely
    - Info: adds "collision", "min_obstacle_dist" keys for TensorBoard

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment.
    n_obstacles : int
        Number of obstacles to place each episode. Default 5.
    obstacle_radius : float
        Radius of each obstacle sphere (m). Default 0.4.
    collision_penalty : float
        Reward penalty per collision step. Default 10.0.
    min_obstacle_dist : float
        Minimum spawn distance between obstacle and goal/AUV to prevent
        impossible starting states. Default 1.5m.
    """

    N_RANGEFINDERS = 4  # forward, port, starboard, up

    def __init__(
        self,
        env,
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

        # Safely extend the observation space bounds
        base_obs = self.env.observation_space
        low = np.append(base_obs.low, np.full(self.N_RANGEFINDERS, -1.0))
        high = np.append(base_obs.high, np.full(self.N_RANGEFINDERS, 20.0))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Cache obstacle geometry IDs for fast lookup during step()
        self._obs_geom_ids = []
        for i in range(n_obstacles):
            try:
                gid = mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}"
                )
                self._obs_geom_ids.append(gid)
            except Exception:
                print(f"Warning: Obstacle obs_{i} not found in XML.")
                pass

        # Find the AUV's main body geom ID to check for contacts
        self._auv_geom_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "halcyon"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Resets base env and scatters obstacles."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Randomise obstacle positions safely
        self._randomise_obstacles()

        # Get extended 23-dim observation
        obs = self._get_extended_obs(obs)
        info["collision"] = False
        info["min_obstacle_dist"] = self._min_obstacle_dist()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advances simulation and checks for catastrophic collisions."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check for physical contact with any obstacle
        collision = self._check_collision()
        if collision:
            reward -= self.collision_penalty
            terminated = True  # End episode immediately on crash

        obs = self._get_extended_obs(obs)
        info["collision"] = collision
        info["min_obstacle_dist"] = self._min_obstacle_dist()

        return obs, reward, terminated, truncated, info

    def _randomise_obstacles(self):
        """Place obstacles at random positions, strictly avoiding AUV start and goal."""
        auv_pos = np.zeros(3)  # AUV always starts at origin (0,0,0)
        goal_pos = self.env._goal_pos

        placed = 0
        attempts = 0

        for gid in self._obs_geom_ids:
            while attempts < 200:
                attempts += 1

                # Sample a random position in spherical coordinates within the workspace
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

                # Keep obstacles submerged but off the seafloor
                pos[2] = np.clip(pos[2], -6.0, 6.0)

                # Professional Polish: Clearance Validation
                # Ensure we don't spawn an obstacle inside the AUV or blocking the goal
                if (
                    np.linalg.norm(pos - auv_pos) > self.min_obstacle_dist
                    and np.linalg.norm(pos - goal_pos) > self.min_obstacle_dist
                ):
                    self.env.model.geom_pos[gid] = pos
                    placed += 1
                    break

        mujoco.mj_forward(self.env.model, self.env.data)

    def _check_collision(self) -> bool:
        """Returns True if the AUV geometry is in contact with any obstacle geometry."""
        for i in range(self.env.data.ncon):
            con = self.env.data.contact[i]
            g1, g2 = con.geom1, con.geom2

            # Check if one of the contact geometries is the AUV and the other is an obstacle
            if (g1 == self._auv_geom_id and g2 in self._obs_geom_ids) or (
                g2 == self._auv_geom_id and g1 in self._obs_geom_ids
            ):
                return True
        return False

    def _min_obstacle_dist(self) -> float:
        """Calculates the distance to the closest obstacle (useful for debugging/logging)."""
        auv_pos = self.env.auv_position
        min_dist = float("inf")
        for gid in self._obs_geom_ids:
            obs_pos = self.env.model.geom_pos[gid]
            d = float(np.linalg.norm(auv_pos - obs_pos)) - self.obstacle_radius
            min_dist = min(min_dist, d)
        return max(min_dist, 0.0)

    def _get_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Appends 4 simulated rangefinder readings to the base observation vector."""
        # Rangefinders are located at indices 23-26 in sensordata in the full Halcyon XML
        # If your XML doesn't define rangefinders, we simulate a basic fallback distance
        try:
            rf = self.env.data.sensordata[23:27].copy()
            # Convert missing hits (-1) to a large default distance (10.0m)
            rf = np.where(rf < 0, 10.0, rf)
        except IndexError:
            # Fallback if rangefinder sensors aren't strictly defined in XML
            # Provides a simple radial distance proxy
            dist = self._min_obstacle_dist()
            rf = np.array([dist, dist, dist, dist], dtype=np.float32)

        return np.append(base_obs, rf.astype(np.float32))


if __name__ == "__main__":
    print("Testing ObstacleAUVWrapper...")
    env = ObstacleAUVWrapper(HalcyonAUVEnv(), n_obstacles=5)
    obs, info = env.reset(seed=42)

    print(f"Observation shape: {obs.shape} (Expected: (23,))")
    print(f"Initial Min Obstacle Dist: {info['min_obstacle_dist']:.2f}m")

    collision_occurred = False
    for step in range(100):
        action = env.action_space.sample()  # Random thruster firing
        obs, r, t, tr, info = env.step(action)
        if info["collision"]:
            print(f"Collision detected at step {step}!")
            collision_occurred = True
            break

    if not collision_occurred:
        print("No collisions during random flailing (this is normal).")

    env.close()
    print("Obstacle Env OK!")
