"""
auv_env.py — HalcyonAUVEnv
════════════════════════════════════════════════════════════════════════════════

Custom Gymnasium environment for the Halcyon X4 AUV (auv.xml).
Wraps MuJoCo, applies realistic fluid physics, and implements a dense
reward function suited for SAC training and sim-to-real transfer research.

Usage:
    env = HalcyonAUVEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Compatible with:
    - gymnasium >= 0.29
    - mujoco >= 3.0
    - stable-baselines3 >= 2.0 (use with VecNormalize wrapper)

Install:
    pip install "gymnasium[mujoco]" "stable-baselines3[extra]" mujoco

Fluid physics applied each step (via xfrc_applied, world frame):
    1. Quadratic drag       F_drag = -c_d * v_body * |v_body|  (body → world)
    2. Buoyancy (net)       F_buoy = mass * g * buoyancy_offset (world +Z)
    3. Water current        F_curr = c_d * (v_curr - v_body) * |v_curr - v_body|
    4. Added mass           F_am   = -C_a * m * a_body          (body → world)

Observation space (19-dim float32):
    [0:3]   goal_vec_body   — goal direction in body frame, L2-normalised
    [3]     goal_dist       — distance to goal (clipped to [0, 20])
    [4:7]   lin_vel_body    — linear velocity in body frame (m/s)
    [7:10]  ang_vel_body    — angular velocity in body frame (rad/s)
    [10:13] euler_angles    — roll, pitch, yaw (rad)
    [13:17] prev_action     — action from previous step (4 thrusters)
    [17]    current_speed   — magnitude of water current (m/s)
    [18]    depth           — AUV depth (world Z, clipped to [-20, 20])

Action space (4-dim continuous, Box[-1, 1]):
    [0] thrust_top
    [1] thrust_bot
    [2] thrust_left
    [3] thrust_right
    → Motors apply 20N max per thruster. gear="20 0 0 0 0 0" in XML.

Reward function (dense, shaped for fast convergence):
    r = w_prog   * progress_reward      # main learning signal
      + w_goal   * goal_reached_bonus   # terminal reward
      + w_alive  * alive_reward         # per-step survival bonus
      - w_energy * energy_penalty       # ctrl effort cost
      - w_smooth * smoothness_penalty   # action jerk cost
      - w_orient * orientation_penalty  # keep nose toward goal
      - w_bound  * boundary_penalty     # out-of-bounds penalty

Bugs fixed:
    BUG 1 — Orientation penalty sign corrected.
            Was: -w * (1.0 + cos_theta) → rewarded pointing AWAY from goal.
            Fix: -w * (1.0 - cos_theta) → penalises pointing away from goal.

    BUG 2 — Buoyancy formula corrected.
            Was: total_mass * g * (1.0 + offset) → ~111N upward at offset=0.
            Fix: total_mass * g * offset → 0N at offset=0 (neutrally buoyant).

    BUG 3 — total_mass cached in __init__ (not recomputed every substep).

    BUG 4 — xfrc_applied cleared BEFORE applying forces each substep.

Research notes:
    - xfrc_applied is in WORLD frame. Drag computed in body frame then
      rotated to world frame via data.xmat (rotation matrix).
    - goal body moved by setting model.body_pos[goal_id] then mj_forward.
    - VecNormalize handles obs/reward normalisation — do not normalise here.
    - frame_skip=4 → effective dt = 4 * 0.01 = 0.04s, ~25Hz control rate.
    - max_episode_steps=500 → 20 seconds per episode at 25Hz.
    - gravity=0 in XML. Net buoyancy = mass * g * buoyancy_offset.
      offset=0.0   → neutrally buoyant (zero net vertical force).
      offset=+0.02 → 2% of weight net upward (floats gently).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding


# ─────────────────────────────────────────────────────────────────────────────
# Default physics parameters (nominal / no DR)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHYSICS = {
    "c_drag_axial": 0.08,
    "c_drag_lateral": 0.20,
    "buoyancy_offset": 0.02,
    "current_velocity": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "added_mass_coeff": 0.15,
    "actuator_efficiency": np.ones(4, dtype=np.float32),
    # Sensor noise (Phase 1 addition)
    "pos_noise_std": 0.0,
    "vel_noise_std": 0.0,
    "ang_noise_std": 0.0,
    "imu_noise_std": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REWARD_WEIGHTS = {
    "progress": 10.0,
    "goal": 50.0,
    "alive": 0.1,
    "energy": 0.02,
    "smoothness": 0.05,
    "orientation": 0.5,
    "boundary": 10.0,
}


class HalcyonAUVEnv(Env):
    """
    Gymnasium environment for Halcyon X4 AUV goal-reaching in 3D.

    Observation: 19-dim (see module docstring)
    Action:       4-dim continuous ∈ [-1, 1]
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_path: Optional[str | Path] = None,
        frame_skip: int = 4,
        max_episode_steps: int = 500,
        goal_threshold: float = 0.5,
        workspace_radius: float = 12.0,
        goal_min_dist: float = 2.0,
        goal_max_dist: float = 6.0,
        physics_params: Optional[Dict] = None,
        reward_weights: Optional[Dict] = None,
        render_mode: Optional[str] = None,
        camera_name: str = "track",
        render_width: int = 640,
        render_height: int = 480,
    ):
        super().__init__()

        # ── Resolve XML path ─────────────────────────────────────────────────
        if xml_path is None:
            xml_path = Path(__file__).parent / "auv.xml"
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"auv.xml not found at {self.xml_path}. "
                "Pass xml_path= explicitly or place auv.xml next to auv_env.py."
            )

        # ── Load MuJoCo model ────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._auv_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "halcyon"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self._n_actuators = self.model.nu  # 4

        # BUG 3 FIX: cache total mass once
        self._total_mass = float(
            sum(self.model.body_mass[i] for i in range(self.model.nbody))
        )

        # Sensor data offsets
        self._sensor_pos_adr = 0
        self._sensor_quat_adr = 3
        self._sensor_linvel_adr = 7
        self._sensor_angvel_adr = 10

        # ── Environment parameters ───────────────────────────────────────────
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold
        self.workspace_radius = workspace_radius
        self.goal_min_dist = goal_min_dist
        self.goal_max_dist = goal_max_dist
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        self.render_mode = render_mode
        self.dt = self.frame_skip * self.model.opt.timestep

        # ── Physics parameters ───────────────────────────────────────────────
        self.physics_params = dict(DEFAULT_PHYSICS)
        if physics_params is not None:
            self.physics_params.update(physics_params)
        self.physics_params["current_velocity"] = np.array(
            self.physics_params["current_velocity"], dtype=np.float32
        )
        self.physics_params["actuator_efficiency"] = np.array(
            self.physics_params["actuator_efficiency"], dtype=np.float32
        )

        # ── Reward weights ───────────────────────────────────────────────────
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # ── Spaces ───────────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_actuators,),
            dtype=np.float32,
        )

        # 19-dim observation space:
        # [0:3] goal_vec_body  [3] goal_dist  [4:7] lin_vel  [7:10] ang_vel
        # [10:13] euler  [13:17] prev_action (4)  [17] current_speed  [18] depth
        obs_high = np.array(
            [
                1.0,
                1.0,
                1.0,  # goal_vec_body (unit vector)
                20.0,  # goal_dist (m)
                3.0,
                3.0,
                3.0,  # lin_vel_body (m/s)
                2.0,
                2.0,
                2.0,  # ang_vel_body (rad/s)
                np.pi,
                np.pi / 2,
                np.pi,  # euler angles (rad)
                1.0,
                1.0,
                1.0,
                1.0,  # prev_action (4 thrusters)
                2.0,  # current_speed (m/s)
                20.0,  # depth (m)
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # ── Episode state ────────────────────────────────────────────────────
        self._step_count = 0
        self._prev_dist = 0.0
        self._prev_action = np.zeros(self._n_actuators, dtype=np.float32)
        self._goal_pos = np.zeros(3, dtype=np.float64)

        # ── Renderer ─────────────────────────────────────────────────────────
        self._renderer = None
        self._viewer = None

        # ── RNG ──────────────────────────────────────────────────────────────
        self.np_random: np.random.Generator = np.random.default_rng()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.qpos[0:3] += self.np_random.uniform(-0.3, 0.3, 3)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        quat = _euler_to_quat(0.0, 0.0, yaw)
        self.data.qpos[3:7] = quat

        self._goal_pos = self._sample_goal()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist = self._goal_distance()
        self._prev_action = np.zeros(self._n_actuators, dtype=np.float32)

        return self._get_observation(), self._get_info()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        effective_ctrl = action * self.physics_params["actuator_efficiency"]
        self.data.ctrl[:] = effective_ctrl

        # BUG 4 FIX: clear → apply → step every substep
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0.0
            self._apply_fluid_forces()
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        self._step_count += 1

        curr_dist = self._goal_distance()
        obs = self._get_observation()
        reward, r_info = self._compute_reward(action, curr_dist)
        terminated = self._is_terminated(curr_dist)
        truncated = self._step_count >= self.max_episode_steps

        if terminated and curr_dist < self.goal_threshold:
            reward += self.reward_weights["goal"]
            r_info["goal_bonus"] = self.reward_weights["goal"]
        else:
            r_info["goal_bonus"] = 0.0

        self._prev_dist = curr_dist
        self._prev_action = action.copy()

        info = self._get_info()
        info.update(r_info)
        info["goal_dist"] = float(curr_dist)
        info["step_count"] = self._step_count

        return obs, float(reward), terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: render / close
    # ─────────────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def _render_rgb_array(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.render_height, self.render_width
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Fluid physics
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_fluid_forces(self):
        """Apply all fluid forces to AUV body each simulation substep."""
        p = self.physics_params
        R = self.data.xmat[self._auv_body_id].reshape(3, 3)

        v_world = self.data.sensordata[
            self._sensor_linvel_adr : self._sensor_linvel_adr + 3
        ].copy()
        v_body = R.T @ v_world

        # 1. Quadratic drag
        c = np.array(
            [p["c_drag_axial"], p["c_drag_lateral"], p["c_drag_lateral"]],
            dtype=np.float64,
        )
        F_drag_world = R @ (-c * v_body * np.abs(v_body))

        # 2. Net buoyancy (BUG 2 FIX: offset only, not full gravity term)
        g = 9.81
        F_buoy_world = np.array(
            [0.0, 0.0, self._total_mass * g * p["buoyancy_offset"]],
            dtype=np.float64,
        )

        # 3. Water current drag
        v_current = p["current_velocity"].astype(np.float64)
        v_rel = v_current - v_world
        F_current_world = p["c_drag_lateral"] * v_rel * np.abs(v_rel)

        # 4. Added mass
        accel_body = self.data.sensordata[13:16].copy()
        F_added_mass_world = R @ (
            -(p["added_mass_coeff"] * self._total_mass) * accel_body
        )

        F_total = F_drag_world + F_buoy_world + F_current_world + F_added_mass_world
        self.data.xfrc_applied[self._auv_body_id, 0:3] = F_total
        self.data.xfrc_applied[self._auv_body_id, 3:6] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the 19-dim observation vector (all in body frame where possible).

        Layout:
            [0:3]   goal_vec_body   — unit vector toward goal in body frame
            [3]     goal_dist       — distance to goal (m), clipped [0,20]
            [4:7]   lin_vel_body    — linear velocity in body frame (m/s)
            [7:10]  ang_vel_body    — angular velocity in body frame (rad/s)
            [10:13] euler_angles    — roll, pitch, yaw (rad)
            [13:17] prev_action     — last applied action (4 thrusters)
            [17]    current_speed   — |current_velocity| (m/s)
            [18]    depth           — AUV Z position (world frame)
        """
        R = self.data.xmat[self._auv_body_id].reshape(3, 3)

        auv_pos_world = self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ].copy()

        goal_vec_world = self._goal_pos - auv_pos_world
        goal_dist = float(np.linalg.norm(goal_vec_world))
        goal_dist_c = np.clip(goal_dist, 0.0, 20.0)

        goal_dir_world = (
            (goal_vec_world / goal_dist)
            if goal_dist > 1e-6
            else np.array([1.0, 0.0, 0.0])
        )
        goal_vec_body = R.T @ goal_dir_world

        lin_vel_world = self.data.sensordata[
            self._sensor_linvel_adr : self._sensor_linvel_adr + 3
        ].copy()
        lin_vel_body = R.T @ lin_vel_world

        ang_vel_world = self.data.sensordata[
            self._sensor_angvel_adr : self._sensor_angvel_adr + 3
        ].copy()
        ang_vel_body = R.T @ ang_vel_world

        quat = self.data.sensordata[
            self._sensor_quat_adr : self._sensor_quat_adr + 4
        ].copy()
        euler = _quat_to_euler(quat)

        current_speed = float(np.linalg.norm(self.physics_params["current_velocity"]))
        depth = float(auv_pos_world[2])

        # Sensor noise (Phase 1)
        p = self.physics_params
        if p.get("pos_noise_std", 0.0) > 0:
            goal_dist_c = float(
                np.clip(
                    goal_dist_c + self.np_random.normal(0, p["pos_noise_std"]),
                    0.0,
                    20.0,
                )
            )
        if p.get("vel_noise_std", 0.0) > 0:
            lin_vel_body = np.clip(
                lin_vel_body + self.np_random.normal(0, p["vel_noise_std"], 3),
                -3.0,
                3.0,
            )
        if p.get("ang_noise_std", 0.0) > 0:
            ang_vel_body = np.clip(
                ang_vel_body + self.np_random.normal(0, p["ang_noise_std"], 3),
                -2.0,
                2.0,
            )

        obs = np.array(
            [
                *goal_vec_body,
                goal_dist_c,
                *np.clip(lin_vel_body, -3.0, 3.0),
                *np.clip(ang_vel_body, -2.0, 2.0),
                *np.clip(euler, -np.pi, np.pi),
                *self._prev_action,  # 4 values → indices 13-16
                current_speed,  # index 17
                np.clip(depth, -20.0, 20.0),  # index 18
            ],
            dtype=np.float32,
        )
        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # Reward
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self, action: np.ndarray, curr_dist: float
    ) -> Tuple[float, Dict]:
        w = self.reward_weights

        progress = self._prev_dist - curr_dist
        r_progress = w["progress"] * np.clip(progress, -1.0, 1.0)
        r_alive = w["alive"]
        r_energy = -w["energy"] * float(np.sum(action**2))
        r_smooth = -w["smoothness"] * float(np.sum((action - self._prev_action) ** 2))

        # BUG 1 FIX: orientation penalty = (1 - cos θ)
        auv_pos_world = self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ]
        goal_vec_world = self._goal_pos - auv_pos_world

        if curr_dist > 1e-6:
            R = self.data.xmat[self._auv_body_id].reshape(3, 3)
            goal_dir_world = goal_vec_world / curr_dist
            goal_dir_body = R.T @ goal_dir_world
            cos_theta = float(goal_dir_body[0])
            r_orient = -w["orientation"] * (1.0 - cos_theta)
        else:
            r_orient = 0.0

        auv_dist = float(np.linalg.norm(auv_pos_world))
        r_boundary = -w["boundary"] if auv_dist > self.workspace_radius else 0.0

        total = r_progress + r_alive + r_energy + r_smooth + r_orient + r_boundary

        return total, {
            "r_progress": float(r_progress),
            "r_alive": float(r_alive),
            "r_energy": float(r_energy),
            "r_smooth": float(r_smooth),
            "r_orient": float(r_orient),
            "r_boundary": float(r_boundary),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _is_terminated(self, curr_dist: float) -> bool:
        if curr_dist < self.goal_threshold:
            return True
        auv_pos = self.data.sensordata[self._sensor_pos_adr : self._sensor_pos_adr + 3]
        if float(np.linalg.norm(auv_pos)) > self.workspace_radius:
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Domain Randomisation hook
    # ─────────────────────────────────────────────────────────────────────────

    def randomize_physics(
        self,
        rng: np.random.Generator,
        drag_range: Tuple[float, float] = (0.1, 0.5),
        buoyancy_range: Tuple[float, float] = (-0.05, 0.10),
        current_speed_range: Tuple[float, float] = (0.0, 0.3),
        added_mass_range: Tuple[float, float] = (0.05, 0.3),
        efficiency_range: Tuple[float, float] = (0.85, 1.0),
    ) -> Dict:
        c_drag_axial = float(rng.uniform(*drag_range)) * 0.4
        c_drag_lateral = float(rng.uniform(*drag_range))
        self.physics_params["c_drag_axial"] = c_drag_axial
        self.physics_params["c_drag_lateral"] = c_drag_lateral

        buoyancy = float(rng.uniform(*buoyancy_range))
        self.physics_params["buoyancy_offset"] = buoyancy

        speed = float(rng.uniform(*current_speed_range))
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        self.physics_params["current_velocity"] = (direction * speed).astype(np.float32)

        self.physics_params["added_mass_coeff"] = float(rng.uniform(*added_mass_range))
        self.physics_params["actuator_efficiency"] = rng.uniform(
            efficiency_range[0], efficiency_range[1], size=4
        ).astype(np.float32)

        return {
            "dr/c_drag_axial": c_drag_axial,
            "dr/c_drag_lateral": c_drag_lateral,
            "dr/buoyancy": buoyancy,
            "dr/current_speed": speed,
            "dr/added_mass": self.physics_params["added_mass_coeff"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_goal(self) -> np.ndarray:
        direction = self.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = self.np_random.uniform(self.goal_min_dist, self.goal_max_dist)
        goal = direction * dist
        goal[2] = np.clip(goal[2], -8.0, 8.0)
        return goal.astype(np.float64)

    def _set_goal_body(self, pos: np.ndarray):
        self.model.body_pos[self._goal_body_id] = pos

    def _goal_distance(self) -> float:
        auv_pos = self.data.sensordata[self._sensor_pos_adr : self._sensor_pos_adr + 3]
        return float(np.linalg.norm(self._goal_pos - auv_pos))

    def _get_info(self) -> Dict:
        auv_pos = self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ].copy()
        lin_vel = self.data.sensordata[
            self._sensor_linvel_adr : self._sensor_linvel_adr + 3
        ].copy()
        return {
            "auv_pos": auv_pos.tolist(),
            "goal_pos": self._goal_pos.tolist(),
            "speed": float(np.linalg.norm(lin_vel)),
            "goal_dist": self._goal_distance(),
            "current_velocity": self.physics_params["current_velocity"].tolist(),
            "drag_axial": self.physics_params["c_drag_axial"],
            "drag_lateral": self.physics_params["c_drag_lateral"],
            "buoyancy_offset": self.physics_params["buoyancy_offset"],
        }

    @property
    def effective_dt(self) -> float:
        return self.dt

    @property
    def goal_position(self) -> np.ndarray:
        return self._goal_pos.copy()

    @property
    def auv_position(self) -> np.ndarray:
        return self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / Euler utilities
# ─────────────────────────────────────────────────────────────────────────────


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium registration
# ─────────────────────────────────────────────────────────────────────────────


def register_env():
    import gymnasium

    gymnasium.register(
        id="HalcyonAUV-v0",
        entry_point="auv_env:HalcyonAUVEnv",
        max_episode_steps=500,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time

    print("=" * 60)
    print("HalcyonAUVEnv — sanity check")
    print("=" * 60)

    xml_candidates = [
        Path(__file__).parent / "auv.xml",
        Path("~/rl_robotics/envs/auv.xml").expanduser(),
        Path("auv.xml"),
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        print("ERROR: auv.xml not found.")
        sys.exit(1)
    print(f"Using XML: {xml_path}")

    env = HalcyonAUVEnv(xml_path=xml_path)
    print(f"Total mass: {env._total_mass:.3f} kg")

    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("✓ SB3 check_env passed")
    except ImportError:
        print("  (SB3 not installed)")

    obs, info = env.reset(seed=42)
    print(f"obs shape: {obs.shape}  (expect (19,))")
    assert obs.shape == (19,), f"FAIL: expected (19,) got {obs.shape}"
    assert env.observation_space.shape == (19,), "FAIL: obs_space mismatch"
    print(f"obs_space: {env.observation_space.shape}  ✓ consistent")
    print(f"Initial goal dist: {info['goal_dist']:.2f}m")

    # Speed test
    t0 = time.perf_counter()
    N = 1000
    total_r = 0.0
    obs, _ = env.reset()
    for _ in range(N):
        a = env.action_space.sample()
        obs, r, t, tr, _ = env.step(a)
        total_r += r
        if t or tr:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0
    print(f"{N} steps in {elapsed:.2f}s → {N / elapsed:.0f} env-steps/sec")
    print(f"Cumulative reward: {total_r:.2f}")

    print("\n✓ All checks passed.")
    env.close()
