"""
auv_moving_goal.py — Moving goal wrapper for HalcyonAUVEnv
===========================================================
Wraps HalcyonAUVEnv so the goal drifts slowly during each episode.
The goal moves at a constant velocity in a random direction,
bouncing off workspace boundaries.

Observation space: same as base env (19-dim). Goal position communicated
via info dict only — not appended to observation.

Usage:
    from auv_env import HalcyonAUVEnv
    from auv_moving_goal import MovingGoalWrapper

    env = MovingGoalWrapper(HalcyonAUVEnv(), goal_speed=0.3)
    obs, info = env.reset()
    # obs.shape == (19,)  — same as base env
    # info["goal_pos"]    — current goal position
    # info["goal_speed"]  — configured goal speed
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

    Observation space is UNCHANGED from the base env (19-dim).
    The goal_vec_body component of the observation automatically reflects
    the updated goal position because _get_observation() is called fresh
    each step after the goal moves.

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment to wrap.
    goal_speed : float
        Speed of goal movement in m/s. Default 0.3.
        0.0 = static (same as base env).
        0.3 = slow drift (realistic AUV target speed).
        0.8 = fast moving target (harder task).
    workspace_radius : float
        Boundary for goal reflection. Should match env.workspace_radius.
        Default 12.0m (matches HalcyonAUVEnv default).
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
        # observation_space is intentionally NOT overridden here.
        # MovingGoalWrapper keeps the same 19-dim obs as base env.

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Sample random goal velocity direction
        direction = self.env.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        direction[2] *= (
            0.3  # mostly horizontal — AUV targets don't move fast vertically
        )
        direction /= np.linalg.norm(direction) + 1e-8

        self._goal_velocity = direction * self.goal_speed

        # Add moving-goal metadata to info
        info["goal_pos"] = self.env._goal_pos.tolist()
        info["goal_speed"] = float(self.goal_speed)
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Move goal one control step
        self._update_goal_position()

        # Recompute observation with updated goal position.
        # This is the key call — goal_vec_body in obs now reflects new goal pos.
        obs = self.env._get_observation()

        info["goal_pos"] = self.env._goal_pos.tolist()
        info["goal_speed"] = float(self.goal_speed)

        return obs, reward, terminated, truncated, info

    def _update_goal_position(self):
        """Move goal by one control step. Reflect off workspace boundary."""
        dt = self.env.effective_dt
        new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Reflect off workspace sphere (80% of radius to keep goal accessible)
        dist = np.linalg.norm(new_pos)
        if dist > self.workspace_radius * 0.8:
            normal = new_pos / (dist + 1e-8)
            self._goal_velocity -= 2.0 * np.dot(self._goal_velocity, normal) * normal
            new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Clamp Z to ocean volume
        new_pos[2] = np.clip(new_pos[2], -8.0, 8.0)

        # Update MuJoCo goal marker
        self.env._goal_pos = new_pos
        self.env._set_goal_body(new_pos)
        mujoco.mj_forward(self.env.model, self.env.data)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────


def make_moving_goal_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    goal_speed: float = 0.3,
):
    """
    Create a moving goal env with DR wrapper.
    Stack: HalcyonAUVEnv → MovingGoalWrapper → AUVDomainRandomWrapper

    Usage:
        env = make_moving_goal_env("envs/auv.xml", mode="curriculum", seed=0)
        obs, info = env.reset()
        # obs.shape == (19,)
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from auv_env import HalcyonAUVEnv
    from auv_dr_wrapper import AUVDomainRandomWrapper

    base = HalcyonAUVEnv(xml_path=xml_path)
    moving = MovingGoalWrapper(base, goal_speed=goal_speed)
    return AUVDomainRandomWrapper(moving, mode=mode, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from auv_env import HalcyonAUVEnv

    print("MovingGoalWrapper — sanity check")

    env = MovingGoalWrapper(HalcyonAUVEnv(), goal_speed=0.5)

    # Check observation space unchanged
    assert env.observation_space.shape == (19,), (
        f"obs_space should be (19,), got {env.observation_space.shape}"
    )
    print(f"✓ obs_space: {env.observation_space.shape}  (matches base env)")

    obs, info = env.reset(seed=0)
    assert obs.shape == (19,), f"obs shape should be (19,), got {obs.shape}"
    print(f"✓ obs shape: {obs.shape}")
    assert "goal_pos" in info, "goal_pos missing from info"
    assert "goal_speed" in info, "goal_speed missing from info"
    print(f"✓ goal_pos in info: {info['goal_pos']}")
    print(f"✓ goal_speed in info: {info['goal_speed']}")

    pos0 = np.array(info["goal_pos"])
    for _ in range(30):
        obs, r, t, tr, info = env.step(env.action_space.sample())
        if t or tr:
            break
    pos1 = np.array(info["goal_pos"])
    dist = np.linalg.norm(pos1 - pos0)

    assert dist > 0.01, f"Goal didn't move: dist={dist:.4f}"
    print(f"✓ Goal moved {dist:.3f}m during episode")
    assert obs.shape == (19,), f"obs shape after step should be (19,), got {obs.shape}"
    print(f"✓ obs shape after step: {obs.shape}")

    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("✓ SB3 check_env passed")
    except ImportError:
        print("  (SB3 not installed)")

    env.close()
    print("\n✓ MovingGoalWrapper all checks passed.")
