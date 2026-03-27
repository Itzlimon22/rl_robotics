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
    2. Buoyancy             F_buoy = +mass * g * (1 + offset)  (world +Z)
    3. Water current        F_curr = c_d * (v_curr - v_body) * |v_curr - v_body|
    4. Added mass (opt.)    F_am   = -C_a * a_body             (body → world)

Observation space (18-dim float32, all normalised to reasonable ranges):
    [0:3]   goal_vec_body   — goal direction in body frame, L2-normalised
    [3]     goal_dist       — distance to goal (clipped to [0, 20])
    [4:7]   lin_vel_body    — linear velocity in body frame (m/s)
    [7:10]  ang_vel_body    — angular velocity in body frame (rad/s)
    [10:13] euler_angles    — roll, pitch, yaw (rad)
    [13:16] prev_action     — action from previous step (for smoothness penalty)
    [16]    current_speed   — magnitude of water current (m/s)
    [17]    depth           — AUV depth below surface (negative = below)

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

Domain randomisation hook (for DR wrapper in Step 3):
    env.physics_params dict is read every step.
    Call env.randomize_physics(rng) to sample new params without reset.

Research notes:
    - xfrc_applied is in WORLD frame. Drag is computed in body frame then
      rotated to world frame via data.xmat (rotation matrix).
    - goal body is moved by setting model.body_pos[goal_id] then mj_forward.
    - VecNormalize handles obs/reward normalisation — do not normalise here.
    - frame_skip=4 → effective dt = 4 * 0.01 = 0.04s, ~25Hz control rate.
    - max_episode_steps=500 → 20 seconds per episode at 25Hz.
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
# These are the values used during training with NO domain randomisation.
# The DR wrapper overrides these each episode reset.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHYSICS = {
    # Quadratic drag coefficient (kg/m).
    # F_drag = -c_drag * v * |v| applied along each axis independently.
    # Real AUV range: 0.1 (streamlined) to 0.8 (high-drag test).
    "c_drag_axial": 0.08,  # along body X (fore-aft, low drag)
    "c_drag_lateral": 0.20,  # along body Y/Z (cross-flow, higher drag)
    # Buoyancy offset as fraction of total weight.
    # 0.0 = neutrally buoyant, +0.02 = 2% positive (floats up slightly).
    "buoyancy_offset": 0.02,
    # Water current vector in world frame (m/s). Zero = still water.
    "current_velocity": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    # Added mass coefficient (fraction of body mass).
    # Virtual mass the AUV must accelerate when it moves through water.
    # Typically 0.1–0.3 for torpedo-shaped bodies.
    "added_mass_coeff": 0.15,
    # Actuator efficiency scaling (1.0 = nominal, <1 = degraded thruster).
    # Applied as a multiplier on data.ctrl before physics step.
    "actuator_efficiency": np.ones(4, dtype=np.float32),
}


# ─────────────────────────────────────────────────────────────────────────────
# Reward weights (tuned for SAC convergence on goal-reaching task)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REWARD_WEIGHTS = {
    "progress": 10.0,  # reward for closing distance to goal each step
    "goal": 50.0,  # bonus on reaching the goal
    "alive": 0.1,  # small per-step survival bonus (encourages longer episodes)
    "energy": 0.02,  # penalise ctrl^2 (encourages efficient thrust use)
    "smoothness": 0.05,  # penalise |action - prev_action| (reduces jitter)
    "orientation": 0.5,  # penalise not facing the goal (encourages nose-toward-goal)
    "boundary": 5.0,  # penalty for leaving the workspace
}


class HalcyonAUVEnv(Env):
    """
    Gymnasium environment for Halcyon X4 AUV goal-reaching in 3D.

    Parameters
    ----------
    xml_path : str | Path | None
        Path to auv.xml. If None, looks in same directory as this file.
    frame_skip : int
        Number of MuJoCo steps per env.step() call. Effective dt = frame_skip * 0.01s.
        Default 4 → 0.04s control period (25 Hz).
    max_episode_steps : int
        Episode truncation length in steps. Default 500 → 20 seconds.
    goal_threshold : float
        Distance (m) at which goal is considered reached. Default 0.5m.
    workspace_radius : float
        Sphere radius (m) — episode ends if AUV exits. Default 15m.
    goal_min_dist : float
        Minimum initial distance to goal at reset. Default 3m.
    goal_max_dist : float
        Maximum initial distance to goal at reset. Default 10m.
    physics_params : dict | None
        Override default physics parameters. None = use DEFAULT_PHYSICS.
    reward_weights : dict | None
        Override default reward weights. None = use DEFAULT_REWARD_WEIGHTS.
    render_mode : str | None
        "human" for live viewer, "rgb_array" for image array, None for no render.
    camera_name : str
        MuJoCo camera name to use for rgb_array render. Default "track".
    render_width : int
        Width of rgb_array render. Default 640.
    render_height : int
        Height of rgb_array render. Default 480.
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
        workspace_radius: float = 15.0,
        goal_min_dist: float = 3.0,
        goal_max_dist: float = 10.0,
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

        # Cache body/actuator/sensor IDs for fast lookup in step()
        self._auv_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "halcyon"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self._n_actuators = self.model.nu  # 4

        # Sensor data offsets (matches sensor layout documented in auv.xml)
        # pos[0:3], quat[3:7], linvel[7:10], angvel[10:13]
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

        # Effective timestep for physics (frame_skip * model timestep)
        self.dt = self.frame_skip * self.model.opt.timestep

        # ── Physics parameters (DR hook) ─────────────────────────────────────
        self.physics_params = dict(DEFAULT_PHYSICS)
        if physics_params is not None:
            self.physics_params.update(physics_params)
        # Make mutable copies of array params
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
        # Action: 4 thrusters, each ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_actuators,),
            dtype=np.float32,
        )

        # Observation: 18 floats (see module docstring for layout)
        # Bounds are loose — VecNormalize will normalise during training.
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
                1.0,  # prev_action
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

        # ── Renderer (initialised lazily on first render call) ───────────────
        self._renderer = None
        self._viewer = None

        # ── RNG (set by reset(seed=...)) ─────────────────────────────────────
        self.np_random: np.random.Generator = np.random.default_rng()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset to a new episode.
        - AUV starts at origin with small random velocity perturbation.
        - Goal placed at a random position in workspace sphere.
        - Physics params unchanged (DR wrapper calls randomize_physics() here).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset MuJoCo to keyframe "home" (origin, zero vel, level attitude)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Small random initial perturbation (avoids degenerate start states)
        self.data.qpos[0:3] += self.np_random.uniform(-0.3, 0.3, 3)
        # Random initial yaw (agent must learn to rotate to face goal)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        quat = _euler_to_quat(0.0, 0.0, yaw)
        self.data.qpos[3:7] = quat

        # Random goal position
        self._goal_pos = self._sample_goal()
        self._set_goal_body(self._goal_pos)

        # Forward pass to propagate state
        mujoco.mj_forward(self.model, self.data)

        # Reset episode tracking
        self._step_count = 0
        self._prev_dist = self._goal_distance()
        self._prev_action = np.zeros(self._n_actuators, dtype=np.float32)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action and advance simulation by frame_skip steps.

        Returns
        -------
        obs          : np.ndarray (18,)
        reward       : float
        terminated   : bool  — goal reached or boundary exceeded
        truncated    : bool  — max_episode_steps reached
        info         : dict  — diagnostic info (all reward components, etc.)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Apply actuator efficiency (DR can degrade individual thrusters)
        effective_ctrl = action * self.physics_params["actuator_efficiency"]
        self.data.ctrl[:] = effective_ctrl

        # Simulate frame_skip steps, applying fluid forces each substep
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0.0
            self._apply_fluid_forces()
            mujoco.mj_step(self.model, self.data)

        # Clear external forces (must reset each frame, not cumulative)
        self.data.xfrc_applied[:] = 0.0

        self._step_count += 1

        # Compute state
        curr_dist = self._goal_distance()
        obs = self._get_observation()
        reward, r_info = self._compute_reward(action, curr_dist)
        terminated = self._is_terminated(curr_dist)
        truncated = self._step_count >= self.max_episode_steps

        # Add terminal goal bonus
        if terminated and curr_dist < self.goal_threshold:
            reward += self.reward_weights["goal"]
            r_info["goal_bonus"] = self.reward_weights["goal"]
        else:
            r_info["goal_bonus"] = 0.0

        # Update trackers
        self._prev_dist = curr_dist
        self._prev_action = action.copy()

        info = self._get_info()
        info.update(r_info)
        info["goal_dist"] = float(curr_dist)
        info["step_count"] = self._step_count

        return obs, float(reward), terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API: render
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
        """
        Apply all fluid forces to the AUV body each simulation substep.

        Forces are computed in BODY frame then rotated to WORLD frame
        before writing to data.xfrc_applied[body_id, :].

        xfrc_applied layout per body: [fx, fy, fz, tx, ty, tz] in world frame.

        Critical: xfrc_applied[i] is ZEROED after each mj_step call,
        so we set it fresh every substep (this is correct behaviour).
        """
        p = self.physics_params

        # ── Get AUV rotation matrix (body → world) ───────────────────────────
        # data.xmat[body_id] is a flattened 3x3 rotation matrix (row-major).
        R = self.data.xmat[self._auv_body_id].reshape(3, 3)  # world = R @ body

        # ── Get velocities in WORLD frame (from sensors) ─────────────────────
        v_world = self.data.sensordata[
            self._sensor_linvel_adr : self._sensor_linvel_adr + 3
        ].copy()

        # ── Transform linear velocity to BODY frame ──────────────────────────
        # v_body = R^T @ v_world  (R is orthonormal so R^T = R^{-1})
        v_body = R.T @ v_world

        # ── 1. Quadratic drag (body frame) ────────────────────────────────────
        # F_drag_body[i] = -c_i * v_body[i] * |v_body[i]|
        # Axial drag (X axis) is lower than lateral drag (Y, Z axes).
        c = np.array(
            [
                p["c_drag_axial"],
                p["c_drag_lateral"],
                p["c_drag_lateral"],
            ],
            dtype=np.float64,
        )
        F_drag_body = -c * v_body * np.abs(v_body)

        # Rotate drag force to world frame
        F_drag_world = R @ F_drag_body

        # ── 2. Buoyancy force (world frame, +Z) ──────────────────────────────
        # gravity = 0 in XML. We apply net buoyancy = mass * g * (1 + offset).
        # With offset=0.02: AUV experiences 2% net upward force (slightly positive).
        total_mass = sum(self.model.body_mass[i] for i in range(self.model.nbody))
        g = 9.81  # m/s^2
        F_buoy_world = np.array(
            [0.0, 0.0, total_mass * g * (1.0 + p["buoyancy_offset"])], dtype=np.float64
        )

        # ── 3. Water current drag (world frame) ───────────────────────────────
        # The current exerts a drag force proportional to relative velocity.
        # F_curr = c_lateral * (v_current - v_AUV) * |v_current - v_AUV|
        # This correctly pushes the AUV in the current direction when stationary
        # and reduces drag when moving with the current.
        v_current = p["current_velocity"].astype(np.float64)
        v_rel = v_current - v_world
        # Use lateral drag coefficient for current force (current hits broad side)
        F_current_world = p["c_drag_lateral"] * v_rel * np.abs(v_rel)

        # ── 4. Added mass (body frame, optional) ──────────────────────────────
        # Virtual inertia from accelerating water. F_am = -C_a * m * a_body.
        # We approximate acceleration as (v - v_prev) / dt using sensor data.
        # Note: this is a first-order approximation; good enough for DR purposes.
        accel_body = self.data.sensordata[13:16].copy()  # accelerometer at IMU
        F_added_mass_body = -(p["added_mass_coeff"] * total_mass) * accel_body
        F_added_mass_world = R @ F_added_mass_body

        # ── Sum all forces and apply ──────────────────────────────────────────
        F_total = F_drag_world + F_buoy_world + F_current_world + F_added_mass_world

        # xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]
        # No external torques from fluid (fins provide passive stability)
        self.data.xfrc_applied[self._auv_body_id, 0:3] = F_total
        self.data.xfrc_applied[self._auv_body_id, 3:6] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the 18-dim observation vector.

        All quantities are expressed in body frame where possible.
        Body-frame observations are invariant to world position/heading,
        which is exactly what we want for a transferable policy.

        Obs layout (matches observation_space):
            [0:3]   goal_vec_body   — unit vector toward goal in body frame
            [3]     goal_dist       — distance to goal (m), clipped [0,20]
            [4:7]   lin_vel_body    — linear velocity in body frame (m/s)
            [7:10]  ang_vel_body    — angular velocity in body frame (rad/s)
            [10:13] euler_angles    — roll, pitch, yaw (rad)
            [13:16] prev_action     — last applied action
            [16]    current_speed   — |current_velocity| (m/s)
            [17]    depth           — AUV Z position (world frame, not body)
        """
        # Rotation matrix body → world
        R = self.data.xmat[self._auv_body_id].reshape(3, 3)

        # AUV position in world
        auv_pos_world = self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ].copy()

        # Goal vector in world frame, then rotate to body frame
        goal_vec_world = self._goal_pos - auv_pos_world
        goal_dist = float(np.linalg.norm(goal_vec_world))
        goal_dist_clipped = np.clip(goal_dist, 0.0, 20.0)

        if goal_dist > 1e-6:
            goal_dir_world = goal_vec_world / goal_dist
        else:
            goal_dir_world = np.array([1.0, 0.0, 0.0])

        goal_vec_body = R.T @ goal_dir_world  # unit vector toward goal in body frame

        # Linear velocity (world → body)
        lin_vel_world = self.data.sensordata[
            self._sensor_linvel_adr : self._sensor_linvel_adr + 3
        ].copy()
        lin_vel_body = R.T @ lin_vel_world

        # Angular velocity (world → body)
        ang_vel_world = self.data.sensordata[
            self._sensor_angvel_adr : self._sensor_angvel_adr + 3
        ].copy()
        ang_vel_body = R.T @ ang_vel_world

        # Euler angles from quaternion (roll, pitch, yaw)
        quat = self.data.sensordata[
            self._sensor_quat_adr : self._sensor_quat_adr + 4
        ].copy()
        euler = _quat_to_euler(quat)  # [roll, pitch, yaw]

        # Current speed (scalar observable — agent can observe current magnitude)
        current_speed = float(np.linalg.norm(self.physics_params["current_velocity"]))

        # Depth (world Z position — AUV should stay within workspace)
        depth = float(auv_pos_world[2])

        obs = np.array(
            [
                *goal_vec_body,  # [0:3]
                goal_dist_clipped,  # [3]
                *np.clip(lin_vel_body, -3.0, 3.0),  # [4:7]
                *np.clip(ang_vel_body, -2.0, 2.0),  # [7:10]
                *np.clip(euler, -np.pi, np.pi),  # [10:13]
                *self._prev_action,  # [13:16]
                current_speed,  # [16]
                np.clip(depth, -20.0, 20.0),  # [17]
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
        """
        Compute shaped reward. Returns (total_reward, component_dict).

        Reward design rationale:
        - Progress reward is the PRIMARY learning signal. It's signed
          (positive = getting closer, negative = getting farther).
          Scaled by 10 to dominate other terms.
        - Energy penalty prevents the policy from using max thrust always.
          Critically important for real AUVs (battery life).
        - Smoothness penalty prevents high-frequency action oscillation
          that would stress actuators and break sim-to-real.
        - Orientation penalty gently encourages nose-toward-goal for
          energy efficiency (axial drag < lateral drag).
        - Boundary penalty is a hard signal — policy learns to stay in.
        """
        w = self.reward_weights

        # ── Progress reward ─────────────────────────────────────────────────
        # Positive when closer, negative when farther. The 1/dt normalisation
        # makes the reward independent of frame_skip choice.
        progress = self._prev_dist - curr_dist
        r_progress = w["progress"] * np.clip(
            progress, -1.0, 1.0
        )  # metres closed this step

        # ── Alive reward ─────────────────────────────────────────────────────
        r_alive = w["alive"]

        # ── Energy penalty ───────────────────────────────────────────────────
        # Penalise sum of squared actuator outputs.
        r_energy = -w["energy"] * float(np.sum(action**2))

        # ── Smoothness penalty ───────────────────────────────────────────────
        # Penalise action change between consecutive steps.
        action_delta = action - self._prev_action
        r_smooth = -w["smoothness"] * float(np.sum(action_delta**2))

        # ── Orientation penalty ──────────────────────────────────────────────
        # Penalise when AUV nose is not pointing toward goal.
        # goal_vec_body[0] = X-component of goal direction in body frame.
        # When nose points at goal, this is 1.0.
        auv_pos_world = self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ]
        goal_vec_world = self._goal_pos - auv_pos_world
        if curr_dist > 1e-6:
            R = self.data.xmat[self._auv_body_id].reshape(3, 3)
            goal_dir_world = goal_vec_world / curr_dist

            # THE BUG FIX: Extract scalar safely with .item()
            goal_dir_body = R.T @ goal_dir_world
            goal_dir_body_x = goal_dir_body[0].item()  # cosine of angle to goal

            # penalty = 1 - cos(θ): 0 when aligned, 2 when pointing away
            r_orient = -w["orientation"] * (1.0 - goal_dir_body_x)
        else:
            r_orient = 0.0

        # ── Boundary penalty ─────────────────────────────────────────────────
        auv_dist_from_origin = float(np.linalg.norm(auv_pos_world))
        if auv_dist_from_origin > self.workspace_radius:
            r_boundary = -w["boundary"]
        else:
            r_boundary = 0.0

        total = r_progress + r_alive + r_energy + r_smooth + r_orient + r_boundary

        components = {
            "r_progress": float(r_progress),
            "r_alive": float(r_alive),
            "r_energy": float(r_energy),
            "r_smooth": float(r_smooth),
            "r_orient": float(r_orient),
            "r_boundary": float(r_boundary),
        }
        return total, components

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _is_terminated(self, curr_dist: float) -> bool:
        """
        Episode terminates (terminated=True, not truncated) if:
          1. AUV reaches goal (curr_dist < goal_threshold), or
          2. AUV exits the workspace sphere (out-of-bounds).
        """
        # Goal reached
        if curr_dist < self.goal_threshold:
            return True

        # Out of bounds
        auv_pos = self.data.sensordata[self._sensor_pos_adr : self._sensor_pos_adr + 3]
        if float(np.linalg.norm(auv_pos)) > self.workspace_radius:
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Domain Randomisation hook (called by DR wrapper in Step 3)
    # ─────────────────────────────────────────────────────────────────────────

    def randomize_physics(
        self,
        rng: np.random.Generator,
        drag_range: Tuple[float, float] = (0.1, 0.5),
        buoyancy_range: Tuple[float, float] = (-0.05, 0.05),
        current_speed_range: Tuple[float, float] = (0.0, 0.3),
        added_mass_range: Tuple[float, float] = (0.05, 0.3),
        efficiency_range: Tuple[float, float] = (0.85, 1.0),
    ) -> Dict:
        """
        Sample new physics parameters from given ranges.

        Called by the DR wrapper at each episode reset.
        Ranges here correspond to TRAINING distribution.
        Test distribution uses wider ranges (see DR wrapper).

        Parameters can also be set directly: env.physics_params["c_drag_axial"] = 0.3

        Returns the sampled params dict (for logging to TensorBoard).
        """
        # Drag coefficients
        c_drag_axial = float(rng.uniform(*drag_range)) * 0.4  # axial is ~40% of lateral
        c_drag_lateral = float(rng.uniform(*drag_range))
        self.physics_params["c_drag_axial"] = c_drag_axial
        self.physics_params["c_drag_lateral"] = c_drag_lateral

        # Buoyancy offset
        buoyancy = float(rng.uniform(*buoyancy_range))
        self.physics_params["buoyancy_offset"] = buoyancy

        # Water current: random direction + random speed
        speed = float(rng.uniform(*current_speed_range))
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        self.physics_params["current_velocity"] = (direction * speed).astype(np.float32)

        # Added mass
        self.physics_params["added_mass_coeff"] = float(rng.uniform(*added_mass_range))

        # Actuator efficiency (each thruster can degrade independently)
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
        """
        Sample a random goal position inside the workspace sphere.
        Goal is at least goal_min_dist and at most goal_max_dist from AUV start.
        """
        # Uniform random point on unit sphere × random radius
        direction = self.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = self.np_random.uniform(self.goal_min_dist, self.goal_max_dist)
        goal = direction * dist
        # Clamp Z to [-8, 8] — keep goal within ocean volume
        goal[2] = np.clip(goal[2], -8.0, 8.0)
        return goal.astype(np.float64)

    def _set_goal_body(self, pos: np.ndarray):
        """Move goal body to pos. Must call mj_forward after to propagate."""
        self.model.body_pos[self._goal_body_id] = pos
        # Note: mj_forward is called in reset() after this.

    def _goal_distance(self) -> float:
        """Return Euclidean distance from AUV to goal."""
        auv_pos = self.data.sensordata[self._sensor_pos_adr : self._sensor_pos_adr + 3]
        return float(np.linalg.norm(self._goal_pos - auv_pos))

    def _get_info(self) -> Dict:
        """Return diagnostic info dict (not used by SAC, useful for logging)."""
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
        """Control period in seconds."""
        return self.dt

    @property
    def goal_position(self) -> np.ndarray:
        """Current goal position (world frame)."""
        return self._goal_pos.copy()

    @property
    def auv_position(self) -> np.ndarray:
        """Current AUV position (world frame)."""
        return self.data.sensordata[
            self._sensor_pos_adr : self._sensor_pos_adr + 3
        ].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / Euler utilities
# ─────────────────────────────────────────────────────────────────────────────


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
    Uses ZYX convention (yaw-pitch-roll intrinsic).
    Returns angles in radians, range [-pi, pi].
    """
    w, x, y, z = quat

    # Roll (rotation around X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotation around Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to MuJoCo quaternion [w, x, y, z].
    Uses ZYX convention.
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium registration
# ─────────────────────────────────────────────────────────────────────────────


def register_env():
    """Register HalcyonAUV-v0 with Gymnasium. Call once at startup."""
    import gymnasium

    gymnasium.register(
        id="HalcyonAUV-v0",
        entry_point="auv_env:HalcyonAUVEnv",
        max_episode_steps=500,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation (run: python auv_env.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    print("=" * 60)
    print("HalcyonAUVEnv — sanity check")
    print("=" * 60)

    # Find auv.xml
    xml_candidates = [
        Path(__file__).parent / "auv.xml",
        Path("~/rl_robotics/envs/auv.xml").expanduser(),
        Path("auv.xml"),
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        print(
            "ERROR: auv.xml not found. Place it next to auv_env.py or in ~/rl_robotics/envs/"
        )
        sys.exit(1)
    print(f"Using XML: {xml_path}")

    env = HalcyonAUVEnv(xml_path=xml_path)

    # SB3 env checker
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("✓ SB3 check_env passed")
    except ImportError:
        print("  (stable-baselines3 not installed, skipping check_env)")

    # Basic rollout
    obs, info = env.reset(seed=42)
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Effective dt:      {env.effective_dt:.3f}s ({1 / env.effective_dt:.1f} Hz)")
    print(f"obs shape:         {obs.shape}  dtype={obs.dtype}")
    print(f"obs (first reset): {np.round(obs, 3)}")
    print(f"Initial goal dist: {info['goal_dist']:.2f}m")
    print()

    # Speed test: 1000 steps
    obs, info = env.reset()
    t0 = time.perf_counter()
    N = 1000
    total_reward = 0.0
    for i in range(N):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    elapsed = time.perf_counter() - t0
    fps = N * env.frame_skip / elapsed

    print(f"Speed test: {N} steps in {elapsed:.2f}s")
    print(f"  → {fps:.0f} sim-steps/sec  ({N / elapsed:.0f} env-steps/sec)")
    print(f"  → Cumulative reward over {N} env-steps: {total_reward:.2f}")
    print()

    # DR test
    rng = np.random.default_rng(0)
    env.reset()
    sampled = env.randomize_physics(rng)
    print("DR test — sampled physics:")
    for k, v in sampled.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("✓ All checks passed. Ready for Step 3 (DR wrapper).")
    env.close()
