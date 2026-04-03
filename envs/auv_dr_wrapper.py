"""
auv_dr_wrapper.py — AUVDomainRandomWrapper
================================================================================
Domain Randomisation wrapper for HalcyonAUVEnv.
Supports hierarchical stacking (e.g., Tracking -> Obstacles -> DR).

This version restores all factory functions and logging callbacks required
by train.py and train_master.py.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

from auv_env import HalcyonAUVEnv


# ══════════════════════════════════════════════════════════════════════════════
# Physics parameter configuration
# ══════════════════════════════════════════════════════════════════════════════

PARAM_CONFIG: Dict[str, Dict[str, float]] = {
    "c_drag_lateral": dict(min_lo=0.10, start_lo=0.15, start_hi=0.25, max_hi=0.50),
    "c_drag_axial": dict(min_lo=0.04, start_lo=0.06, start_hi=0.10, max_hi=0.20),
    "buoyancy_offset": dict(min_lo=-0.05, start_lo=-0.01, start_hi=0.05, max_hi=0.10),
    "current_speed": dict(min_lo=0.00, start_lo=0.00, start_hi=0.10, max_hi=0.30),
    "added_mass": dict(min_lo=0.05, start_lo=0.10, start_hi=0.20, max_hi=0.30),
    "act_efficiency": dict(min_lo=0.80, start_lo=0.95, start_hi=1.00, max_hi=1.00),
    # Sensor noise parameters (Phase 1)
    "pos_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.02, max_hi=0.10),
    "vel_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.01, max_hi=0.05),
    "ang_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.005, max_hi=0.02),
}

TEST_PARAM_CONFIG: Dict[str, Tuple[float, float]] = {
    "c_drag_lateral": (0.30, 0.80),
    "c_drag_axial": (0.12, 0.32),
    "buoyancy_offset": (-0.10, 0.15),
    "current_speed": (0.20, 0.60),
    "added_mass": (0.20, 0.50),
    "act_efficiency": (0.70, 1.00),
    "pos_noise_std": (0.05, 0.15),
    "vel_noise_std": (0.02, 0.08),
    "ang_noise_std": (0.01, 0.04),
}

CDR_WINDOW_SIZE = 20
CDR_EXPAND_THRESHOLD = 0.40
CDR_CONTRACT_THRESHOLD = 0.20
CDR_EXPAND_STEP = 0.05
CDR_CONTRACT_STEP = 0.03

# ══════════════════════════════════════════════════════════════════════════════
# Main Wrapper
# ══════════════════════════════════════════════════════════════════════════════


class AUVDomainRandomWrapper(Wrapper):
    """
    Domain Randomisation wrapper for HalcyonAUVEnv.
    Supports 'none', 'uniform', and 'curriculum' modes.
    """

    VALID_MODES = ("none", "uniform", "curriculum")

    def __init__(
        self,
        env: gym.Env,
        mode: str = "curriculum",
        seed: Optional[int] = None,
        cdr_window_size: int = CDR_WINDOW_SIZE,
        cdr_expand_threshold: float = CDR_EXPAND_THRESHOLD,
        cdr_contract_threshold: float = CDR_CONTRACT_THRESHOLD,
        cdr_expand_step: float = CDR_EXPAND_STEP,
        cdr_contract_step: float = CDR_CONTRACT_STEP,
        verbose: bool = True,
    ):
        super().__init__(env)

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self._cdr_window_size = cdr_window_size
        self._cdr_expand_threshold = cdr_expand_threshold
        self._cdr_contract_threshold = cdr_contract_threshold
        self._cdr_expand_step = cdr_expand_step
        self._cdr_contract_step = cdr_contract_step

        self._outcome_window: Deque[bool] = deque(maxlen=cdr_window_size)
        self._cdr_ranges: Dict[str, List[float]] = {
            k: [v["start_lo"], v["start_hi"]] for k, v in PARAM_CONFIG.items()
        }

        self._curriculum_level: float = 0.0
        self._n_episodes: int = 0
        self._n_successes: int = 0
        self._last_episode_was_success: bool = False
        self._last_dr_params: Dict[str, float] = {}
        self._test_mode: bool = False

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        if self._n_episodes > 0:
            self._record_episode_outcome()
            if self.mode == "curriculum":
                self._maybe_update_curriculum()

        self._last_dr_params = self._apply_dr()
        self._n_episodes += 1
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self._dr_info())
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            goal_dist = info.get("goal_dist", float("inf"))
            # Prioritize 'is_success' from tracking/obstacle wrappers
            self._last_episode_was_success = info.get(
                "is_success",
                terminated and goal_dist < self.env.unwrapped.goal_threshold,
            )
            # Catastrophic failure if collision occurred
            if info.get("collision", False):
                self._last_episode_was_success = False

        info.update(self._dr_info())
        return obs, reward, terminated, truncated, info

    def _apply_dr(self) -> Dict[str, float]:
        if self._test_mode:
            return self._sample_and_apply(TEST_PARAM_CONFIG)
        if self.mode == "none":
            return self._apply_nominal()
        if self.mode == "uniform":
            ranges = {k: (v["min_lo"], v["max_hi"]) for k, v in PARAM_CONFIG.items()}
            return self._sample_and_apply(ranges)
        if self.mode == "curriculum":
            ranges = {k: (lo, hi) for k, (lo, hi) in self._cdr_ranges.items()}
            return self._sample_and_apply(ranges)
        return {}

    def _apply_nominal(self) -> Dict[str, float]:
        """Reach down to core physics and return nominal values."""
        core = self.env.unwrapped
        p = core.physics_params
        return {
            "c_drag_lateral": p["c_drag_lateral"],
            "c_drag_axial": p["c_drag_axial"],
            "buoyancy": p["buoyancy_offset"],
            "current_speed": float(np.linalg.norm(p["current_velocity"])),
            "added_mass": p["added_mass_coeff"],
            "act_efficiency": float(np.mean(p["actuator_efficiency"])),
        }

    def _sample_and_apply(
        self, ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Sample physics parameters and apply to the .unwrapped core environment."""
        r = self.rng
        core = self.env.unwrapped
        p = core.physics_params

        # -- Drag --
        c_dl = float(r.uniform(*ranges["c_drag_lateral"]))
        c_da = float(r.uniform(*ranges["c_drag_axial"]))
        p["c_drag_lateral"] = c_dl
        p["c_drag_axial"] = c_da

        # -- Buoyancy --
        buoy = float(r.uniform(*ranges["buoyancy_offset"]))
        p["buoyancy_offset"] = buoy

        # -- Water current --
        speed = float(r.uniform(*ranges["current_speed"]))
        direction = r.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        p["current_velocity"] = (direction * speed).astype(np.float32)

        # -- Added mass --
        p["added_mass_coeff"] = float(r.uniform(*ranges["added_mass"]))

        # -- Actuator efficiency --
        eff_lo, eff_hi = ranges["act_efficiency"]
        eff = r.uniform(eff_lo, eff_hi, size=4).astype(np.float32)
        p["actuator_efficiency"] = eff

        # -- Sensor noise (Phase 1) --
        for key in ["pos_noise_std", "vel_noise_std", "ang_noise_std"]:
            if key in ranges:
                p[key] = float(r.uniform(*ranges[key]))

        return {
            "c_drag_lateral": c_dl,
            "c_drag_axial": c_da,
            "buoyancy": buoy,
            "current_speed": speed,
            "added_mass": p["added_mass_coeff"],
            "act_efficiency": float(np.mean(eff)),
            "pos_noise": p.get("pos_noise_std", 0.0),
            "vel_noise": p.get("vel_noise_std", 0.0),
        }

    def _record_episode_outcome(self):
        self._outcome_window.append(self._last_episode_was_success)
        if self._last_episode_was_success:
            self._n_successes += 1

    def _maybe_update_curriculum(self):
        if len(self._outcome_window) < self._cdr_window_size:
            return
        sr = sum(self._outcome_window) / len(self._outcome_window)

        if sr > self._cdr_expand_threshold:
            self._adjust_ranges(self._cdr_expand_step, expand=True)
            if self.verbose:
                print(f"[CDR] ep={self._n_episodes:5d} | SR={sr:.2f} | EXPAND")
        elif sr < self._cdr_contract_threshold:
            self._adjust_ranges(self._cdr_contract_step, expand=False)
            if self.verbose:
                print(f"[CDR] ep={self._n_episodes:5d} | SR={sr:.2f} | CONTRACT")

        self._curriculum_level = self._compute_curriculum_level()

    def _adjust_ranges(self, step_fraction: float, expand: bool):
        for k, cfg in PARAM_CONFIG.items():
            f_lo = cfg["start_lo"] - cfg["min_lo"]
            f_hi = cfg["max_hi"] - cfg["start_hi"]
            lo, hi = self._cdr_ranges[k]

            if expand:
                lo, hi = lo - step_fraction * f_lo, hi + step_fraction * f_hi
            else:
                lo, hi = lo + step_fraction * f_lo, hi - step_fraction * f_hi

            self._cdr_ranges[k] = [
                float(np.clip(lo, cfg["min_lo"], cfg["start_lo"])),
                float(np.clip(hi, cfg["start_hi"], cfg["max_hi"])),
            ]

    def _compute_curriculum_level(self) -> float:
        levels = []
        for k, cfg in PARAM_CONFIG.items():
            lo, hi = self._cdr_ranges[k]
            lo_p = (cfg["start_lo"] - lo) / (cfg["start_lo"] - cfg["min_lo"] + 1e-8)
            hi_p = (hi - cfg["start_hi"]) / (cfg["max_hi"] - cfg["start_hi"] + 1e-8)
            levels.append(np.clip((lo_p + hi_p) / 2.0, 0.0, 1.0))
        return float(np.mean(levels))

    def _dr_info(self) -> Dict[str, Any]:
        info = {"dr/mode": self.mode, "dr/n_episodes": self._n_episodes}
        info.update({f"dr/{k}": v for k, v in self._last_dr_params.items()})
        if self.mode == "curriculum":
            sr = (
                sum(self._outcome_window) / len(self._outcome_window)
                if self._outcome_window
                else 0.0
            )
            info.update(
                {"dr/curriculum_level": self._curriculum_level, "dr/success_rate": sr}
            )
        return info

    def set_test_distribution(self):
        self._test_mode = True

    def unset_test_distribution(self):
        self._test_mode = False

    @property
    def curriculum_level(self) -> float:
        return self._curriculum_level

    @property
    def rolling_success_rate(self) -> float:
        if not self._outcome_window:
            return 0.0
        return sum(self._outcome_window) / len(self._outcome_window)


# ══════════════════════════════════════════════════════════════════════════════
# SB3 Callbacks & Factory Functions (Restored)
# ══════════════════════════════════════════════════════════════════════════════


def make_training_callbacks(
    eval_env,
    log_dir: str,
    eval_freq: int = 10000,
    n_eval_episodes: int = 20,
    save_path: Optional[str] = None,
):
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )

    class CDRLoggingCallback(BaseCallback):
        _MODE_MAP = {"none": 0, "uniform": 1, "curriculum": 2}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", [{}]):
                for k in [
                    "curriculum_level",
                    "success_rate",
                    "current_speed",
                    "c_drag_lateral",
                    "n_episodes",
                ]:
                    if f"dr/{k}" in info:
                        self.logger.record(f"cdr/{k}", info[f"dr/{k}"])
                if "dr/mode" in info:
                    self.logger.record(
                        "cdr/mode", self._MODE_MAP.get(info["dr/mode"], -1)
                    )
                if "goal_dist" in info:
                    self.logger.record("env/goal_dist", info["goal_dist"])
            return True

    callbacks = [CDRLoggingCallback()]
    if save_path:
        callbacks.append(
            CheckpointCallback(
                save_freq=50000, save_path=save_path, name_prefix="auv_model"
            )
        )

    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
    )
    return CallbackList(callbacks)


def make_auv_env(xml_path: str, mode: str = "curriculum", seed: Optional[int] = None):
    """Original factory for static goal task."""
    base = HalcyonAUVEnv(xml_path=xml_path)
    return AUVDomainRandomWrapper(base, mode=mode, seed=seed)


def make_tracking_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    path_speed: float = 0.3,
):
    """Factory for figure-8 tracking task."""
    from auv_tracking_env import HalcyonAUVTrackingEnv

    base = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=path_speed)
    return AUVDomainRandomWrapper(base, mode=mode, seed=seed)


def make_obstacle_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    n_obstacles: int = 5,
):
    """Factory for obstacle avoidance task."""
    from auv_obstacle_env import ObstacleAUVWrapper

    base = HalcyonAUVEnv(xml_path=xml_path)
    obs_env = ObstacleAUVWrapper(base, n_obstacles=n_obstacles)
    return AUVDomainRandomWrapper(obs_env, mode=mode, seed=seed)


def make_master_env(
    xml_path: str, mode: str = "curriculum", seed: Optional[int] = None
):
    """Factory for the Grand Challenge: Tracking + Obstacles."""
    from auv_tracking_env import HalcyonAUVTrackingEnv
    from auv_obstacle_env import ObstacleAUVWrapper

    # FIX: Hardcode 0.3 instead of the undefined path_speed variable
    base = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)
    return AUVDomainRandomWrapper(
        ObstacleAUVWrapper(base, n_obstacles=5), mode=mode, seed=seed
    )
