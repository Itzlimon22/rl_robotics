"""
auv_dr_wrapper.py — AUVDomainRandomWrapper
════════════════════════════════════════════════════════════════════════════════

Domain Randomisation wrapper for HalcyonAUVEnv.
Implements three experimental conditions:

  mode="none"       → No DR. Nominal physics every episode.
                      Paper label: "Naive SAC" (baseline 1)

  mode="uniform"    → Uniform DR. Physics sampled uniformly from
                      full training ranges each episode reset.
                      Paper label: "Uniform DR" (baseline 2)

  mode="curriculum" → Curriculum DR (CDR). Ranges start narrow and
                      expand automatically as agent improves, based
                      on a rolling success rate window.
                      Paper label: "CDR (ours)"

Architecture:
    AUVDomainRandomWrapper
        └── wraps HalcyonAUVEnv (gymnasium.Wrapper)
                └── wraps auv.xml (MuJoCo)

Fixes vs original:
  - _apply_dr() now checks _test_mode → set_test_distribution() works
  - _expand_ranges() / _contract_ranges() clamp with np.clip → no FP drift
  - _apply_nominal() returns act_efficiency → no missing key in logs
  - CDRLoggingCallback logs mode + n_episodes for TensorBoard phase plots
  - Validation __main__ runs all modes + checkpoint + speed test

Install:
    pip install "gymnasium[mujoco]" "stable-baselines3[extra]" mujoco numpy
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
#
# Each entry: (min_lo, start_lo, start_hi, max_hi)
#   min_lo   = minimum lower bound (CDR contracts toward this)
#   start_lo = initial lower bound for CDR (easy start)
#   start_hi = initial upper bound for CDR (easy start)
#   max_hi   = maximum upper bound (CDR expands toward this)
#
# Uniform DR always samples from [min_lo, max_hi].
# CDR starts at [start_lo, start_hi] and expands both bounds.
#
# Paper Table 1 reference:
#                    Nominal   Train start     Train max       Test (held-out)
#   c_drag_lateral:  0.20      [0.15, 0.25]    [0.10, 0.50]   [0.30, 0.80]
#   c_drag_axial:    0.08      [0.06, 0.10]    [0.04, 0.20]   [0.12, 0.32]
#   buoyancy_offset: 0.02      [-0.01, 0.05]   [-0.05, 0.10]  [-0.10, 0.15]
#   current_speed:   0.00      [0.00, 0.10]    [0.00, 0.30]   [0.20, 0.60]
#   added_mass:      0.15      [0.10, 0.20]    [0.05, 0.30]   [0.20, 0.50]
#   act_efficiency:  1.00      [0.95, 1.00]    [0.80, 1.00]   [0.70, 1.00]

PARAM_CONFIG: Dict[str, Dict[str, float]] = {
    "c_drag_lateral": dict(min_lo=0.10, start_lo=0.15, start_hi=0.25, max_hi=0.50),
    "c_drag_axial": dict(min_lo=0.04, start_lo=0.06, start_hi=0.10, max_hi=0.20),
    "buoyancy_offset": dict(min_lo=-0.05, start_lo=-0.01, start_hi=0.05, max_hi=0.10),
    "current_speed": dict(min_lo=0.00, start_lo=0.00, start_hi=0.10, max_hi=0.30),
    "added_mass": dict(min_lo=0.05, start_lo=0.10, start_hi=0.20, max_hi=0.30),
    "act_efficiency": dict(min_lo=0.80, start_lo=0.95, start_hi=1.00, max_hi=1.00),
    # Sensor noise parameters (NEW)
    "pos_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.02, max_hi=0.10),
    "vel_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.01, max_hi=0.05),
    "ang_noise_std": dict(min_lo=0.00, start_lo=0.00, start_hi=0.005, max_hi=0.02),
}

# Held-out test distribution — NEVER seen during training.
# Transfer success on these ranges = main paper result.
TEST_PARAM_CONFIG: Dict[str, Tuple[float, float]] = {
    "c_drag_lateral": (0.30, 0.80),
    "c_drag_axial": (0.12, 0.32),
    "buoyancy_offset": (-0.10, 0.15),
    "current_speed": (0.20, 0.60),
    "added_mass": (0.20, 0.50),
    "act_efficiency": (0.70, 1.00),
    "pos_noise_std": (0.05, 0.15),  # stronger noise in test
    "vel_noise_std": (0.02, 0.08),
    "ang_noise_std": (0.01, 0.04),
}


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

CDR_WINDOW_SIZE = 20  # rolling window for success rate
CDR_EXPAND_THRESHOLD = 0.40  # success rate above this → expand ranges
CDR_CONTRACT_THRESHOLD = 0.20  # success rate below this → contract ranges
CDR_EXPAND_STEP = 0.05  # fraction of full range to expand per trigger
CDR_CONTRACT_STEP = 0.03  # fraction of full range to contract per trigger
CDR_MIN_EPISODES = CDR_WINDOW_SIZE  # warm-up before curriculum adjusts


# ══════════════════════════════════════════════════════════════════════════════
# Main Wrapper
# ══════════════════════════════════════════════════════════════════════════════


class AUVDomainRandomWrapper(Wrapper):
    """
    Domain Randomisation wrapper for HalcyonAUVEnv.

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment to wrap.
    mode : str
        One of "none", "uniform", "curriculum".
    seed : int | None
        RNG seed for reproducibility.
    cdr_window_size : int
        Rolling success rate window length.
    cdr_expand_threshold : float
        Expand ranges when success rate exceeds this.
    cdr_contract_threshold : float
        Contract ranges when success rate drops below this.
    cdr_expand_step : float
        Fraction of full range to expand per trigger.
    cdr_contract_step : float
        Fraction of full range to contract per trigger.
    verbose : bool
        Print curriculum updates to stdout.
    """

    VALID_MODES = ("none", "uniform", "curriculum")

    def __init__(
        self,
        env: HalcyonAUVEnv,
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

        # ── CDR hyperparameters ───────────────────────────────────────────────
        self._cdr_window_size = cdr_window_size
        self._cdr_expand_threshold = cdr_expand_threshold
        self._cdr_contract_threshold = cdr_contract_threshold
        self._cdr_expand_step = cdr_expand_step
        self._cdr_contract_step = cdr_contract_step

        # Rolling window: True=success, False=failure
        self._outcome_window: Deque[bool] = deque(maxlen=cdr_window_size)

        # Current CDR ranges — start narrow, expand as agent improves
        self._cdr_ranges: Dict[str, List[float]] = {
            k: [v["start_lo"], v["start_hi"]] for k, v in PARAM_CONFIG.items()
        }

        # Scalar ∈ [0, 1]: how far the curriculum has expanded
        # 0.0 = start (easy), 1.0 = maximum training difficulty
        self._curriculum_level: float = 0.0

        # ── Episode tracking ──────────────────────────────────────────────────
        self._n_episodes: int = 0
        self._n_successes: int = 0
        self._last_episode_was_success: bool = False
        self._last_dr_params: Dict[str, float] = {}

        # Used to detect episode end inside step()
        self._episode_terminated = False
        self._episode_truncated = False

        # Set to True by set_test_distribution() — switches sampling
        # to the held-out test config (wider than training max)
        self._test_mode: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and apply DR for the new episode.
        CDR curriculum update happens here, using the previous episode outcome.
        """
        # Update curriculum from previous episode (skip on first episode)
        if self._n_episodes > 0:
            self._record_episode_outcome()
            if self.mode == "curriculum":
                self._maybe_update_curriculum()

        # Sample and apply physics for the new episode
        self._last_dr_params = self._apply_dr()
        self._n_episodes += 1
        self._episode_terminated = False
        self._episode_truncated = False

        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self._dr_info())
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Pass-through step. Tracks episode outcome for curriculum update.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._episode_terminated = terminated
            self._episode_truncated = truncated
            goal_dist = info.get("goal_dist", float("inf"))
            # Success = episode terminated by reaching goal, NOT by time limit
            self._last_episode_was_success = (
                terminated and goal_dist < self.env.goal_threshold
            )

        info.update(self._dr_info())
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # DR application
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_dr(self) -> Dict[str, float]:
        """
        Route to correct DR method based on mode and _test_mode flag.

        BUG FIX (vs original): _test_mode is now checked here, so calling
        set_test_distribution() actually routes to _apply_test().
        Previously _test_mode was set but never read.
        """
        if self._test_mode:
            return self._apply_test()
        if self.mode == "none":
            return self._apply_nominal()
        if self.mode == "uniform":
            return self._apply_uniform()
        if self.mode == "curriculum":
            return self._apply_curriculum()
        return {}

    def _apply_nominal(self) -> Dict[str, float]:
        """
        Mode 'none': leave physics at DEFAULT_PHYSICS, return current values.
        No randomize_physics() call — env keeps whatever was set at init.

        BUG FIX (vs original): now also returns act_efficiency so the
        _last_dr_params dict has no missing keys in _dr_info().
        """
        return {
            "c_drag_lateral": self.env.physics_params["c_drag_lateral"],
            "c_drag_axial": self.env.physics_params["c_drag_axial"],
            "buoyancy": self.env.physics_params["buoyancy_offset"],
            "current_speed": float(
                np.linalg.norm(self.env.physics_params["current_velocity"])
            ),
            "added_mass": self.env.physics_params["added_mass_coeff"],
            "act_efficiency": float(
                np.mean(self.env.physics_params["actuator_efficiency"])
            ),
        }

    def _apply_uniform(self) -> Dict[str, float]:
        """
        Mode 'uniform': sample uniformly from full training ranges [min_lo, max_hi].
        Standard Uniform Domain Randomization baseline (Tobin et al. 2017).
        """
        ranges = {k: (v["min_lo"], v["max_hi"]) for k, v in PARAM_CONFIG.items()}
        return self._sample_and_apply(ranges)

    def _apply_curriculum(self) -> Dict[str, float]:
        """
        Mode 'curriculum': sample from current CDR ranges [lo, hi],
        which start narrow at [start_lo, start_hi] and expand as agent improves.
        """
        ranges = {k: (lo, hi) for k, (lo, hi) in self._cdr_ranges.items()}
        return self._sample_and_apply(ranges)

    def _apply_test(self) -> Dict[str, float]:
        """
        Sample from held-out test distribution (wider than any training range).
        Used for zero-shot transfer evaluation (main paper result).
        """
        return self._sample_and_apply(TEST_PARAM_CONFIG)

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling helper
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_and_apply(
        self, ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Sample physics parameters from given ranges and apply to base env.

        Parameters
        ----------
        ranges : dict of {param_name: (lo, hi)}

        Returns
        -------
        dict of sampled float values (for logging).
        """
        r = self.rng

        # ── Drag ─────────────────────────────────────────────────────────────
        c_dl = float(r.uniform(*ranges["c_drag_lateral"]))
        c_da = float(r.uniform(*ranges["c_drag_axial"]))
        self.env.physics_params["c_drag_lateral"] = c_dl
        self.env.physics_params["c_drag_axial"] = c_da

        # ── Buoyancy ──────────────────────────────────────────────────────────
        buoy = float(r.uniform(*ranges["buoyancy_offset"]))
        self.env.physics_params["buoyancy_offset"] = buoy

        # ── Water current (random speed + random direction) ───────────────────
        speed = float(r.uniform(*ranges["current_speed"]))
        direction = r.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        self.env.physics_params["current_velocity"] = (direction * speed).astype(
            np.float32
        )

        # ── Added mass ────────────────────────────────────────────────────────
        am = float(r.uniform(*ranges["added_mass"]))
        self.env.physics_params["added_mass_coeff"] = am

        # ── Actuator efficiency (per-thruster) ────────────────────────────────
        eff_lo, eff_hi = ranges["act_efficiency"]
        eff = r.uniform(eff_lo, eff_hi, size=4).astype(np.float32)
        self.env.physics_params["actuator_efficiency"] = eff

        # Sensor noise (NEW)
        if "pos_noise_std" in ranges:
            self.env.physics_params["pos_noise_std"] = float(
                r.uniform(*ranges["pos_noise_std"])
            )
        if "vel_noise_std" in ranges:
            self.env.physics_params["vel_noise_std"] = float(
                r.uniform(*ranges["vel_noise_std"])
            )
        if "ang_noise_std" in ranges:
            self.env.physics_params["ang_noise_std"] = float(
                r.uniform(*ranges["ang_noise_std"])
            )

        return {
            "c_drag_lateral": c_dl,
            "c_drag_axial": c_da,
            "buoyancy": buoy,
            "current_speed": speed,
            "added_mass": am,
            "act_efficiency": float(np.mean(eff)),
            "pos_noise": self.env.physics_params.get("pos_noise_std", 0.0),
            "vel_noise": self.env.physics_params.get("vel_noise_std", 0.0),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Curriculum logic
    # ─────────────────────────────────────────────────────────────────────────

    def _record_episode_outcome(self):
        """Record last episode success/failure into the rolling window."""
        success = self._last_episode_was_success
        self._outcome_window.append(success)
        if success:
            self._n_successes += 1

    def _maybe_update_curriculum(self):
        """
        Check rolling success rate and expand or contract CDR ranges.
        Warm-up: no adjustments until the window is fully populated.
        """
        if len(self._outcome_window) < self._cdr_window_size:
            return

        success_rate = sum(self._outcome_window) / len(self._outcome_window)

        if success_rate > self._cdr_expand_threshold:
            self._expand_ranges()
            if self.verbose:
                print(
                    f"[CDR] ep={self._n_episodes:5d} | "
                    f"success={success_rate:.2f} > {self._cdr_expand_threshold:.2f} "
                    f"→ EXPAND  | level={self._curriculum_level:.3f}"
                )
        elif success_rate < self._cdr_contract_threshold:
            self._contract_ranges()
            if self.verbose:
                print(
                    f"[CDR] ep={self._n_episodes:5d} | "
                    f"success={success_rate:.2f} < {self._cdr_contract_threshold:.2f} "
                    f"→ CONTRACT | level={self._curriculum_level:.3f}"
                )

        self._curriculum_level = self._compute_curriculum_level()

    def _expand_ranges(self):
        """
        Expand all CDR ranges toward their maximum bounds by cdr_expand_step.

        Each parameter's lo moves toward min_lo and hi moves toward max_hi,
        each by expand_step × full_range. Parameters with larger absolute
        ranges expand faster in absolute terms but at the same relative rate.

        BUG FIX: np.clip() applied after expansion to prevent floating-point
        drift from pushing bounds outside their allowed limits over many
        expand/contract cycles.
        """
        for k, cfg in PARAM_CONFIG.items():
            full_lo = cfg["start_lo"] - cfg["min_lo"]  # max downward movement for lo
            full_hi = cfg["max_hi"] - cfg["start_hi"]  # max upward movement for hi
            step_lo = self._cdr_expand_step * full_lo
            step_hi = self._cdr_expand_step * full_hi

            lo, hi = self._cdr_ranges[k]
            lo = lo - step_lo  # move lo down toward min_lo
            hi = hi + step_hi  # move hi up toward max_hi

            # BUG FIX: clamp to prevent drift beyond hard limits
            lo = np.clip(lo, cfg["min_lo"], cfg["start_lo"])
            hi = np.clip(hi, cfg["start_hi"], cfg["max_hi"])

            self._cdr_ranges[k] = [float(lo), float(hi)]

    def _contract_ranges(self):
        """
        Contract all CDR ranges back toward their starting values.
        Prevents curriculum from staying in territory the agent can't handle.

        BUG FIX: np.clip() applied after contraction, same rationale as expand.
        """
        for k, cfg in PARAM_CONFIG.items():
            full_lo = cfg["start_lo"] - cfg["min_lo"]
            full_hi = cfg["max_hi"] - cfg["start_hi"]
            step_lo = self._cdr_contract_step * full_lo
            step_hi = self._cdr_contract_step * full_hi

            lo, hi = self._cdr_ranges[k]
            lo = lo + step_lo  # move lo up toward start_lo
            hi = hi - step_hi  # move hi down toward start_hi

            # BUG FIX: clamp to prevent over-contraction
            lo = np.clip(lo, cfg["min_lo"], cfg["start_lo"])
            hi = np.clip(hi, cfg["start_hi"], cfg["max_hi"])

            self._cdr_ranges[k] = [float(lo), float(hi)]

    def _compute_curriculum_level(self) -> float:
        """
        Compute scalar curriculum level in [0, 1].
        0.0 = all params at start (easy). 1.0 = all params at max difficulty.
        Computed as average fractional expansion across all parameters.
        """
        levels = []
        for k, cfg in PARAM_CONFIG.items():
            lo, hi = self._cdr_ranges[k]

            lo_range = cfg["start_lo"] - cfg["min_lo"]
            hi_range = cfg["max_hi"] - cfg["start_hi"]

            lo_prog = (cfg["start_lo"] - lo) / (lo_range + 1e-8)
            hi_prog = (hi - cfg["start_hi"]) / (hi_range + 1e-8)

            levels.append(float(np.clip((lo_prog + hi_prog) / 2.0, 0.0, 1.0)))

        return float(np.mean(levels))

    # ─────────────────────────────────────────────────────────────────────────
    # Info / TensorBoard logging
    # ─────────────────────────────────────────────────────────────────────────

    def _dr_info(self) -> Dict[str, Any]:
        """
        Build DR-related info dict returned on every step().
        Keys prefixed dr/ for clean TensorBoard filtering.
        """
        info: Dict[str, Any] = {
            "dr/mode": self.mode,
            "dr/n_episodes": self._n_episodes,
        }
        info.update({f"dr/{k}": v for k, v in self._last_dr_params.items()})

        if self.mode == "curriculum":
            sr = (
                sum(self._outcome_window) / len(self._outcome_window)
                if self._outcome_window
                else 0.0
            )
            info["dr/curriculum_level"] = self._curriculum_level
            info["dr/success_rate"] = sr
            for k, (lo, hi) in self._cdr_ranges.items():
                info[f"dr/range_{k}_lo"] = lo
                info[f"dr/range_{k}_hi"] = hi

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def set_test_distribution(self):
        """
        Switch DR to the held-out test distribution.
        Call this on the eval env before transfer evaluation.

        Usage:
            eval_env = AUVDomainRandomWrapper(HalcyonAUVEnv(...), mode="uniform")
            eval_env.set_test_distribution()
            # eval_env now samples from TEST_PARAM_CONFIG (wider ranges)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=50)
        """
        self._test_mode = True

    def unset_test_distribution(self):
        """Revert to normal training distribution."""
        self._test_mode = False

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def curriculum_level(self) -> float:
        """Scalar curriculum level ∈ [0, 1]. Only meaningful in CDR mode."""
        return self._curriculum_level

    @property
    def rolling_success_rate(self) -> float:
        """Rolling success rate over last window. CDR mode only."""
        if not self._outcome_window:
            return 0.0
        return sum(self._outcome_window) / len(self._outcome_window)

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def n_successes(self) -> int:
        return self._n_successes

    # ─────────────────────────────────────────────────────────────────────────
    # CDR checkpoint / restore
    # ─────────────────────────────────────────────────────────────────────────

    def get_cdr_state(self) -> Dict:
        """
        Return current CDR state as a JSON-serialisable dict.
        Save alongside model weights to resume training exactly.

        Usage:
            import json
            state = env.get_cdr_state()
            json.dump(state, open("cdr_checkpoint.json", "w"), indent=2)
        """
        return {
            "mode": self.mode,
            "n_episodes": self._n_episodes,
            "n_successes": self._n_successes,
            "curriculum_level": self._curriculum_level,
            "cdr_ranges": {k: list(v) for k, v in self._cdr_ranges.items()},
            "outcome_window": list(self._outcome_window),
        }

    def load_cdr_state(self, state: Dict):
        """
        Restore CDR state from a checkpoint dict.

        Usage:
            import json
            state = json.load(open("cdr_checkpoint.json"))
            env.load_cdr_state(state)
        """
        self._n_episodes = state["n_episodes"]
        self._n_successes = state["n_successes"]
        self._curriculum_level = state["curriculum_level"]
        self._cdr_ranges = {k: list(v) for k, v in state["cdr_ranges"].items()}
        self._outcome_window = deque(
            state["outcome_window"], maxlen=self._cdr_window_size
        )


# ══════════════════════════════════════════════════════════════════════════════
# SB3 callbacks
# ══════════════════════════════════════════════════════════════════════════════


def make_training_callbacks(
    eval_env,
    log_dir: str,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
    save_path: Optional[str] = None,
):
    """
    Build SB3 CallbackList for AUV CDR training.

    Includes:
      - CDRLoggingCallback  : logs curriculum_level, success_rate,
                              drag, current, mode to TensorBoard every step
      - CheckpointCallback  : saves model every 50k steps (if save_path given)
      - EvalCallback        : evaluates on held-out env every eval_freq steps

    Parameters
    ----------
    eval_env : gymnasium env (wrapped with AUVDomainRandomWrapper)
        Evaluation environment. Call eval_env.set_test_distribution() before
        passing here to evaluate on the held-out test distribution.
    log_dir : str
        Directory for TensorBoard logs and best model.
    eval_freq : int
        Evaluate every N training steps.
    n_eval_episodes : int
        Number of episodes per evaluation.
    save_path : str | None
        Directory for model checkpoints. None = skip checkpointing.

    Usage:
        callbacks = make_training_callbacks(
            eval_env=eval_env,
            log_dir="~/rl_research/auv/cdr_run1/",
            eval_freq=10_000,
            n_eval_episodes=20,
            save_path="~/rl_research/auv/cdr_run1/checkpoints/",
        )
        model.learn(total_timesteps=1_000_000, callback=callbacks)
    """
    try:
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
    except ImportError:
        raise ImportError(
            "stable-baselines3 required. pip install 'stable-baselines3[extra]'"
        )

    class CDRLoggingCallback(BaseCallback):
        """
        Log CDR metrics to TensorBoard at every step.

        Logged keys (visible in TensorBoard):
          cdr/curriculum_level   — expansion progress [0, 1]
          cdr/rolling_success    — rolling success rate [0, 1]
          cdr/current_speed      — sampled current this episode
          cdr/c_drag_lateral     — sampled lateral drag this episode
          cdr/mode               — 0=none, 1=uniform, 2=curriculum
          cdr/n_episodes         — total episodes elapsed
          env/goal_dist          — current distance to goal

        BUG FIX vs original: mode and n_episodes are now logged as scalars
        for TensorBoard phase plots (useful for comparing training runs).
        """

        _MODE_MAP = {"none": 0, "uniform": 1, "curriculum": 2}

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [{}])
            for info in infos:
                # CDR-specific
                if "dr/curriculum_level" in info:
                    self.logger.record(
                        "cdr/curriculum_level", info["dr/curriculum_level"]
                    )
                if "dr/success_rate" in info:
                    self.logger.record("cdr/rolling_success", info["dr/success_rate"])
                if "dr/current_speed" in info:
                    self.logger.record("cdr/current_speed", info["dr/current_speed"])
                if "dr/c_drag_lateral" in info:
                    self.logger.record("cdr/c_drag_lateral", info["dr/c_drag_lateral"])
                # BUG FIX: log mode as int and n_episodes as scalar
                if "dr/mode" in info:
                    self.logger.record(
                        "cdr/mode", self._MODE_MAP.get(info["dr/mode"], -1)
                    )
                if "dr/n_episodes" in info:
                    self.logger.record("cdr/n_episodes", info["dr/n_episodes"])
                # Environment
                if "goal_dist" in info:
                    self.logger.record("env/goal_dist", info["goal_dist"])
            return True

    callbacks = [CDRLoggingCallback()]

    if save_path:
        callbacks.append(
            CheckpointCallback(
                save_freq=50_000,
                save_path=save_path,
                name_prefix="auv_model",
                verbose=1,
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
            render=False,
            verbose=1,
        )
    )

    return CallbackList(callbacks)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ══════════════════════════════════════════════════════════════════════════════


def make_auv_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> AUVDomainRandomWrapper:
    """
    One-liner factory: create a ready-to-use AUV env with DR wrapper.

    Usage:
        env = make_auv_env("envs/auv.xml", mode="curriculum", seed=42)
        obs, info = env.reset()
    """
    base = HalcyonAUVEnv(xml_path=xml_path, render_mode=render_mode)
    return AUVDomainRandomWrapper(base, mode=mode, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════
# Validation (run: python auv_dr_wrapper.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("AUVDomainRandomWrapper — sanity check")
    print("=" * 60)

    # Locate auv.xml
    candidates = [
        Path(__file__).parent / "auv.xml",
        Path("~/rl_robotics/envs/auv.xml").expanduser(),
        Path("auv.xml"),
    ]
    xml_path = next((p for p in candidates if p.exists()), None)
    if xml_path is None:
        print("ERROR: auv.xml not found.")
        sys.exit(1)
    print(f"Using: {xml_path}\n")

    # ── Test all three modes ──────────────────────────────────────────────────
    for mode in ("none", "uniform", "curriculum"):
        print(f"--- mode='{mode}' ---")
        env = make_auv_env(str(xml_path), mode=mode, seed=0)

        try:
            from stable_baselines3.common.env_checker import check_env

            check_env(env, warn=True)
            print("  ✓ SB3 check_env passed")
        except ImportError:
            print("  (SB3 not installed — skipping check_env)")

        obs, info = env.reset(seed=0)
        n_steps = n_resets = 0
        for _ in range(200):
            obs, r, t, tr, info = env.step(env.action_space.sample())
            n_steps += 1
            if t or tr:
                obs, info = env.reset()
                n_resets += 1

        print(f"  Steps: {n_steps} | Resets: {n_resets}")
        print(f"  DR info keys: {[k for k in info if k.startswith('dr/')]}")

        if mode == "curriculum":
            print(f"  curriculum_level : {env.curriculum_level:.3f}")
            print(f"  rolling_success  : {env.rolling_success_rate:.3f}")
            print(f"  CDR ranges after {env.n_episodes} episodes:")
            for k, (lo, hi) in env._cdr_ranges.items():
                cfg = PARAM_CONFIG[k]
                print(
                    f"    {k:22s}: [{lo:.4f}, {hi:.4f}]"
                    f"  (start=[{cfg['start_lo']:.3f},{cfg['start_hi']:.3f}]"
                    f"  max=[{cfg['min_lo']:.3f},{cfg['max_hi']:.3f}])"
                )

        env.close()
        print()

    # ── Test set_test_distribution ────────────────────────────────────────────
    print("--- set_test_distribution() test ---")
    env = make_auv_env(str(xml_path), mode="uniform", seed=5)
    env.set_test_distribution()
    obs, info = env.reset()
    speed = info.get("dr/current_speed", -1)
    drag = info.get("dr/c_drag_lateral", -1)
    print(f"  current_speed={speed:.3f}  (test range: [0.20, 0.60])")
    print(f"  c_drag_lateral={drag:.3f}  (test range: [0.30, 0.80])")
    assert 0.20 <= speed <= 0.60, f"current_speed {speed} outside test range!"
    assert 0.30 <= drag <= 0.80, f"c_drag_lateral {drag} outside test range!"
    print("  ✓ set_test_distribution() correctly routes to TEST_PARAM_CONFIG")
    env.close()
    print()

    # ── CDR checkpoint / restore ──────────────────────────────────────────────
    print("--- CDR checkpoint / restore test ---")
    env1 = make_auv_env(str(xml_path), mode="curriculum", seed=1)
    env1.reset()
    for _ in range(10):
        _, _, t, tr, _ = env1.step(env1.action_space.sample())
        if t or tr:
            env1.reset()
    state = env1.get_cdr_state()
    print(f"  get_cdr_state() keys: {list(state.keys())}")

    env2 = make_auv_env(str(xml_path), mode="curriculum", seed=99)
    env2.load_cdr_state(state)
    assert env2._n_episodes == env1._n_episodes, "episode count mismatch"
    assert env2._cdr_ranges == env1._cdr_ranges, "cdr_ranges mismatch"
    print("  ✓ CDR state checkpoint / restore OK")
    env1.close()
    env2.close()
    print()

    # ── Curriculum clamp verification (FP drift guard) ────────────────────────
    print("--- Curriculum clamp test (expand × 1000) ---")
    env = make_auv_env(str(xml_path), mode="curriculum", seed=7)
    for _ in range(1000):
        env._expand_ranges()
    for k, cfg in PARAM_CONFIG.items():
        lo, hi = env._cdr_ranges[k]
        assert lo >= cfg["min_lo"] - 1e-9, f"{k} lo={lo} below min_lo={cfg['min_lo']}"
        assert hi <= cfg["max_hi"] + 1e-9, f"{k} hi={hi} above max_hi={cfg['max_hi']}"
    print("  ✓ No FP drift after 1000 expand calls")
    env.close()
    print()

    # ── Speed test ────────────────────────────────────────────────────────────
    print("--- Speed test (mode='curriculum', 1000 steps) ---")
    env = make_auv_env(str(xml_path), mode="curriculum", seed=42)
    env.reset()
    t0 = time.perf_counter()
    N = 1000
    for _ in range(N):
        obs, r, t, tr, _ = env.step(env.action_space.sample())
        if t or tr:
            env.reset()
    elapsed = time.perf_counter() - t0
    print(f"  {N} steps in {elapsed:.2f}s → {N / elapsed:.0f} env-steps/sec")
    env.close()

    print("\n✓ All checks passed. Ready for Step 5: baseline SAC training.\n")
    print("Next commands:")
    print("  from auv_dr_wrapper import make_auv_env, make_training_callbacks")
    print("  env = make_auv_env('envs/auv.xml', mode='curriculum', seed=0)")
