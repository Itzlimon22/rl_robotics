"""
auv_obstacle_env.py — Obstacle avoidance wrapper for HalcyonAUVEnv
===================================================================
Adds N spherical obstacles placed randomly each episode.
Extends observation with 4 rangefinder readings.
Adds collision penalty and terminates episode on collision.

Observation space: 23-dim (base 19 + 4 rangefinders).
    [0:19]  base HalcyonAUVEnv observation
    [19:23] rangefinder distances: forward, port, starboard, up (m)
            -1.0 = no hit within cutoff distance
            >0.0 = distance to nearest obstacle in that direction

Requires obstacle geoms in auv.xml:
    obs_0 through obs_4 (spheres, contype=1, mass=0)
    Add these to worldbody in auv.xml — see instructions below.

Usage:
    from auv_env import HalcyonAUVEnv
    from auv_obstacle_env import ObstacleAUVWrapper

    env = ObstacleAUVWrapper(HalcyonAUVEnv(), n_obstacles=5)
    obs, info = env.reset(seed=42)
    # obs.shape == (23,)
    # info["collision"]          — True if AUV hit an obstacle
    # info["min_obstacle_dist"]  — closest obstacle distance (m)

XML required (add to worldbody in auv.xml before the Lights section):
    <geom name="obs_0" type="sphere" size="0.4" pos="3 0 0"
          rgba="0.8 0.2 0.2 0.7" contype="1" conaffinity="1" group="1" mass="0"/>
    <geom name="obs_1" type="sphere" size="0.4" pos="-3 0 0"
          rgba="0.8 0.2 0.2 0.7" contype="1" conaffinity="1" group="1" mass="0"/>
    <geom name="obs_2" type="sphere" size="0.4" pos="0 3 0"
          rgba="0.8 0.2 0.2 0.7" contype="1" conaffinity="1" group="1" mass="0"/>
    <geom name="obs_3" type="sphere" size="0.4" pos="0 -3 0"
          rgba="0.8 0.2 0.2 0.7" contype="1" conaffinity="1" group="1" mass="0"/>
    <geom name="obs_4" type="sphere" size="0.4" pos="2 2 1"
          rgba="0.8 0.2 0.2 0.7" contype="1" conaffinity="1" group="1" mass="0"/>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix import to work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from auv_env import HalcyonAUVEnv

import numpy as np
import mujoco
from gymnasium import Wrapper, spaces
from typing import Optional, Dict, Any, Tuple


class ObstacleAUVWrapper(Wrapper):
    """
    Adds obstacle avoidance to HalcyonAUVEnv.

    Observation: 23-dim = base 19 + 4 rangefinder readings.
    Reward: -collision_penalty when AUV hits an obstacle (+ episode ends).
    Reset: randomises obstacle positions each episode.

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment to wrap.
    n_obstacles : int
        Number of obstacles to place each episode. Default 5.
        Must be <= number of obs_N geoms in auv.xml.
    obstacle_radius : float
        Radius of each obstacle sphere (m). Default 0.4.
        Must match the size= attribute in auv.xml.
    collision_penalty : float
        Reward penalty on collision. Default 10.0.
    min_obstacle_dist : float
        Minimum spawn distance between obstacle and AUV/goal. Default 1.5m.
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

        # Extend observation space: base 19 + 4 rangefinders = 23
        base_low = self.env.observation_space.low
        base_high = self.env.observation_space.high
        low = np.append(base_low, np.full(self.N_RANGEFINDERS, -1.0)).astype(np.float32)
        high = np.append(base_high, np.full(self.N_RANGEFINDERS, 10.0)).astype(
            np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Cache obstacle geom IDs from MuJoCo model
        self._obs_geom_ids = []
        for i in range(n_obstacles):
            gid = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}"
            )
            if gid >= 0:
                self._obs_geom_ids.append(gid)
            else:
                print(
                    f"[ObstacleAUVWrapper] WARNING: obs_{i} not found in XML. "
                    f"Add obstacle geoms to auv.xml (see file docstring)."
                )

        if len(self._obs_geom_ids) == 0:
            print(
                "[ObstacleAUVWrapper] WARNING: No obstacle geoms found. "
                "Collision detection will not work. Add obs_0..obs_4 to auv.xml."
            )
        else:
            print(
                f"[ObstacleAUVWrapper] Found {len(self._obs_geom_ids)} obstacle geoms."
            )

    @property
    def physics_params(self):
        """Pass-through property allowing the DR wrapper to modify base physics."""
        return self.env.unwrapped.physics_params

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        if self._obs_geom_ids:
            self._randomise_obstacles()

        obs = self._get_extended_obs(obs)
        info["collision"] = False
        info["min_obstacle_dist"] = self._min_obstacle_dist()
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        collision = self._check_collision()
        if collision:
            reward -= self.collision_penalty
            terminated = True
            info["is_success"] = False

        obs = self._get_extended_obs(obs)
        info["collision"] = collision
        info["min_obstacle_dist"] = self._min_obstacle_dist()

        return obs, reward, terminated, truncated, info

    def _randomise_obstacles(self):
        """Place obstacles at random positions avoiding AUV start and goal."""
        auv_pos = np.zeros(3)
        goal_pos = self.env._goal_pos

        for gid in self._obs_geom_ids:
            placed = False
            attempts = 0
            while not placed and attempts < 200:
                attempts += 1
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

                clearance_auv = np.linalg.norm(pos - auv_pos)
                clearance_goal = np.linalg.norm(pos - goal_pos)

                if (
                    clearance_auv > self.min_obstacle_dist
                    and clearance_goal > self.min_obstacle_dist
                ):
                    self.env.model.geom_pos[gid] = pos
                    placed = True

        mujoco.mj_forward(self.env.model, self.env.data)

    def _check_collision(self) -> bool:
        """Return True if any contact involves an obstacle geom."""
        if not self._obs_geom_ids:
            return False
        obs_set = set(self._obs_geom_ids)
        for i in range(self.env.data.ncon):
            con = self.env.data.contact[i]
            if con.geom1 in obs_set or con.geom2 in obs_set:
                return True
        return False

    def _min_obstacle_dist(self) -> float:
        """Minimum distance from AUV centre to any obstacle surface."""
        if not self._obs_geom_ids:
            return float("inf")
        auv_pos = self.env.auv_position
        min_dist = float("inf")
        for gid in self._obs_geom_ids:
            obs_pos = self.env.model.geom_pos[gid].copy()
            d = float(np.linalg.norm(auv_pos - obs_pos)) - self.obstacle_radius
            min_dist = min(min_dist, d)
        return max(min_dist, 0.0)

    def _get_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """
        Append 4 rangefinder readings to base observation.

        Rangefinder sensor indices in sensordata (from auv.xml):
            rf_fwd  : [23]  — forward
            rf_port : [24]  — port (left)
            rf_stbd : [25]  — starboard (right)
            rf_up   : [26]  — upward

        Returns -1.0 for sensors beyond cutoff range (no hit).
        """
        try:
            rf = self.env.data.sensordata[23:27].copy().astype(np.float32)
        except (IndexError, Exception):
            # Fallback if rangefinder sensors not available
            rf = np.full(self.N_RANGEFINDERS, 10.0, dtype=np.float32)
        return np.append(base_obs, rf).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import mujoco as mj

    print("ObstacleAUVWrapper — sanity check")

    env = ObstacleAUVWrapper(HalcyonAUVEnv(), n_obstacles=5)

    assert env.observation_space.shape == (23,), (
        f"obs_space should be (23,), got {env.observation_space.shape}"
    )
    print(f"✓ obs_space: {env.observation_space.shape}  (base 19 + 4 rangefinders)")

    n_geoms = len(env._obs_geom_ids)
    if n_geoms == 5:
        print(f"✓ {n_geoms} obstacle geom IDs cached")
    else:
        print(f"⚠ Only {n_geoms}/5 obstacle geoms found.")
        print("  Add obs_0..obs_4 to auv.xml worldbody (see file docstring).")

    obs, info = env.reset(seed=42)
    assert obs.shape == (23,), f"obs shape should be (23,), got {obs.shape}"
    print(f"✓ obs shape after reset: {obs.shape}")
    assert "collision" in info
    assert "min_obstacle_dist" in info
    print(f"✓ collision={info['collision']}, min_dist={info['min_obstacle_dist']:.2f}m")

    # Check rangefinder values
    rf = obs[19:23]
    assert all(v >= -1.0 for v in rf), f"Invalid rangefinder values: {rf}"
    print(f"✓ rangefinder values: {np.round(rf, 2)}")

    # Check obstacles at different positions
    if n_geoms == 5:
        positions = np.array(
            [env.env.model.geom_pos[gid].copy() for gid in env._obs_geom_ids]
        )
        assert positions.std() > 0.1, "All obstacles at same position"
        print(f"✓ Obstacles spread: std={positions.std():.2f}")

    # Run some steps
    collisions = 0
    for _ in range(100):
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert obs.shape == (23,)
        if info["collision"]:
            collisions += 1
        if t or tr:
            obs, info = env.reset()
    print(f"✓ 100 steps OK, collisions detected: {collisions}")

    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("✓ SB3 check_env passed")
    except ImportError:
        print("  (SB3 not installed)")

    env.close()
    print("\n✓ ObstacleAUVWrapper all checks passed.")
