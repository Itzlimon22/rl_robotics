"""
auv_cdr_env.py — Curriculum Domain Randomization (CDR) Wrapper
================================================================
The novel contribution of:
  "Curriculum Domain Randomization for Sim-to-Real Transfer in AUV Control"

Key idea:
  - Start training with NARROW physics parameter ranges (easy, near-nominal)
  - As the agent improves (tracked via a rolling mean reward), WIDEN the ranges
  - This is a smooth curriculum: no discrete stages, just continuous interpolation
  - Compare against: (a) No DR, (b) Uniform DR (full range from episode 1)

Curriculum stages (5 levels, auto-advance):
  Stage 0 — Nominal:     ranges at 0% of max width (fixed nominal params)
  Stage 1 — Narrow:      ranges at 25% of max width
  Stage 2 — Medium:      ranges at 50% of max width
  Stage 3 — Wide:        ranges at 75% of max width
  Stage 4 — Full:        ranges at 100% of max width (= Uniform DR baseline)

Advancement criterion:
  - Rolling mean reward over last `window` episodes exceeds `threshold`
  - Threshold scales with stage (harder to advance as difficulty increases)

Usage:
    from envs.auv_cdr_env import AUVCurriculumDREnv

    env = AUVCurriculumDREnv(
        nominal_env_kwargs={},   # kwargs passed to AUVDREnv / AUVEnv
        window=50,               # episodes to average reward over
        advance_threshold=0.7,   # fraction of max reward to advance
        verbose=True,
    )
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

# ── Import your existing DR env ──────────────────────────────────────────────
# Adjust the import path if your project structure differs.
try:
    from envs.auv_dr_env import AUVDREnv
except ImportError:
    warnings.warn(
        "Could not import AUVDREnv from envs.auv_dr_env. "
        "Make sure auv_dr_env.py is in ~/rl_robotics/envs/ and your "
        "PYTHONPATH includes ~/rl_robotics/. "
        "Falling back to a stub for development.",
        ImportWarning,
        stacklevel=2,
    )
    AUVDREnv = None  # Handled gracefully in __init__


# ══════════════════════════════════════════════════════════════════════════════
# Physics parameter definitions
# ══════════════════════════════════════════════════════════════════════════════

# Nominal (centre) values — the "true" simulator values.
NOMINAL_PARAMS: Dict[str, float] = {
    "drag_coeff": 0.30,
    "buoyancy_offset": 0.00,
    "current_speed": 0.00,
    # current_direction is sampled uniformly on the sphere; no nominal needed.
}

# Maximum half-widths per parameter (Stage 4 = Uniform DR baseline).
# Training ranges from the research doc:
#   drag:      [0.10, 0.50]  → centre=0.30, half-width=0.20
#   buoyancy:  [-0.05, 0.05] → centre=0.00, half-width=0.05
#   current:   [0.00, 0.30]  → centre=0.15, half-width=0.15
MAX_HALF_WIDTHS: Dict[str, float] = {
    "drag_coeff": 0.20,
    "buoyancy_offset": 0.05,
    "current_speed": 0.15,
}

# Interpolation fractions for each stage (Stage 0 → Stage 4).
STAGE_FRACTIONS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]
NUM_STAGES: int = len(STAGE_FRACTIONS)

# Per-stage rolling-mean-reward thresholds to advance.
# BUG FIX: moved to module-level constant (was duplicated inline in original).
# Tune these values after your first debug run on the real env.
ADVANCE_THRESHOLDS: Dict[int, float] = {
    0: -50.0,  # Nominal → Narrow:  agent navigates reliably at nominal physics
    1: -40.0,  # Narrow  → Medium:  consistent performance under mild DR
    2: -30.0,  # Medium  → Wide:    robust under moderate physics variation
    3: -20.0,  # Wide    → Full:    generalises across the full training range
    # Stage 4 is terminal; no entry needed.
}


# ══════════════════════════════════════════════════════════════════════════════
# CDR Wrapper
# ══════════════════════════════════════════════════════════════════════════════


class AUVCurriculumDREnv(gym.Wrapper):
    """
    Curriculum Domain Randomization wrapper for AUVDREnv.

    Wraps AUVDREnv and overrides the DR parameter ranges at each episode
    reset based on the current curriculum stage.  Stage advances automatically
    when the rolling mean episode reward exceeds the stage threshold.

    Parameters
    ----------
    env_or_kwargs : gym.Env | dict | None
        A pre-built AUVDREnv instance, or a dict of kwargs to construct one.
        Pass ``{}`` or ``None`` to use AUVDREnv defaults.
    window : int
        Number of episodes for the rolling mean reward (default 50).
    advance_thresholds : dict[int, float] | None
        Maps stage index → reward threshold to advance.
        Overrides module-level ADVANCE_THRESHOLDS if provided.
    verbose : bool
        Print stage advancement messages.
    log_interval : int
        Print curriculum stats every N completed episodes.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_or_kwargs: Any = None,
        window: int = 50,
        advance_thresholds: Optional[Dict[int, float]] = None,
        verbose: bool = True,
        log_interval: int = 100,
    ):
        # ── Build or accept the base DR environment ───────────────────────────
        if env_or_kwargs is None or isinstance(env_or_kwargs, dict):
            if AUVDREnv is None:
                raise ImportError(
                    "AUVDREnv could not be imported. "
                    "Check your envs/auv_dr_env.py and PYTHONPATH."
                )
            kwargs = env_or_kwargs or {}
            base_env = AUVDREnv(**kwargs)
        elif isinstance(env_or_kwargs, gym.Env):
            base_env = env_or_kwargs
        else:
            raise TypeError(
                f"env_or_kwargs must be a dict or gym.Env, got {type(env_or_kwargs)}"
            )

        super().__init__(base_env)

        # ── Curriculum state ──────────────────────────────────────────────────
        self.stage: int = 0
        self.episode_count: int = 0
        self.episode_reward: float = 0.0
        self.window: int = window
        self.verbose: bool = verbose
        self.log_interval: int = log_interval

        # BUG FIX: maxlen is set correctly from the `window` param.
        # The original set maxlen=window on the deque but then stored `window`
        # as a separate attribute without ever using it to guard advancement —
        # advancement was guarded by `len(self.reward_history) < self.window`,
        # which was correct, but the deque's maxlen was also `window` so those
        # two were always in sync.  Made explicit here for clarity.
        self.reward_history: deque = deque(maxlen=window)

        self.advance_thresholds: Dict[int, float] = advance_thresholds or dict(
            ADVANCE_THRESHOLDS
        )

        # Research / logging bookkeeping
        self.stage_history: List[Tuple[int, int]] = []  # (episode, new_stage)
        self.mean_reward_at_advance: List[float] = []  # rolling mean at each advance

    # ─────────────────────────────────────────────────────────────────────────
    # Curriculum helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_fraction(self) -> float:
        """DR interpolation fraction for the current stage (0.0 → 1.0)."""
        return STAGE_FRACTIONS[self.stage]

    def _get_current_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute DR parameter ranges for the current curriculum stage.

        Returns
        -------
        dict of {param_name: (low, high)}

        BUG FIX: current_speed lower-bound clamp now uses np.clip instead of
        max() so it is safe when frac=0 (Stage 0 gives half-width=0, meaning
        both bounds equal the nominal 0.0 — no clamping needed, but explicit
        clip is more robust against floating-point rounding).
        """
        frac = self.current_fraction
        ranges: Dict[str, Tuple[float, float]] = {}
        for param, nominal in NOMINAL_PARAMS.items():
            half = MAX_HALF_WIDTHS[param] * frac
            lo = nominal - half
            hi = nominal + half
            ranges[param] = (float(lo), float(hi))

        # current_speed must be non-negative (speed cannot be < 0).
        cs_lo, cs_hi = ranges["current_speed"]
        ranges["current_speed"] = (float(np.clip(cs_lo, 0.0, None)), cs_hi)
        return ranges

    def _maybe_advance_stage(self) -> bool:
        """
        Evaluate whether the rolling mean reward earns a stage advance.

        Returns True if the stage was advanced, False otherwise.

        BUG FIX: the original returned after a single advance even if the
        new stage's threshold was also already met (possible after force_stage
        or a long training pause).  This version loops so the agent can skip
        multiple stages in one check if warranted — prevents a stale easy
        stage from blocking progress.
        """
        advanced = False

        while self.stage < NUM_STAGES - 1:
            if len(self.reward_history) < self.window:
                break  # warm-up: not enough episodes yet

            mean_reward = float(np.mean(self.reward_history))
            threshold = self.advance_thresholds.get(self.stage, float("inf"))

            if mean_reward < threshold:
                break  # not ready to advance

            # ── Advance ───────────────────────────────────────────────────────
            self.stage += 1
            self.mean_reward_at_advance.append(mean_reward)
            self.stage_history.append((self.episode_count, self.stage))
            advanced = True

            if self.verbose:
                frac = self.current_fraction * 100
                print(
                    f"\n[CDR] ✦ Stage advanced → Stage {self.stage} "
                    f"(DR width: {frac:.0f}% of max)  |  "
                    f"Episode {self.episode_count}  |  "
                    f"Mean reward: {mean_reward:.2f}\n"
                )

        return advanced

    def _apply_curriculum_ranges(self):
        """
        Push the current curriculum ranges into the base DR environment.

        Tries ``env.set_dr_ranges(ranges)`` first (clean API), then falls back
        to direct attribute assignment for envs that expose named range attrs.

        BUG FIX: the fallback now also updates ``current_speed_range`` using
        the computed range dict, not a stale hard-coded value.
        """
        ranges = self._get_current_ranges()

        if hasattr(self.env, "set_dr_ranges"):
            self.env.set_dr_ranges(ranges)
            return

        # Fallback: set individual range attributes directly.
        attr_map = {
            "drag_coeff": "drag_range",
            "buoyancy_offset": "buoyancy_range",
            "current_speed": "current_speed_range",
        }
        for param, attr in attr_map.items():
            if hasattr(self.env, attr):
                setattr(self.env, attr, ranges[param])

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply curriculum ranges then delegate to the base env.
        Injects CDR metadata into the returned info dict.
        """
        self._apply_curriculum_ranges()
        obs, info = self.env.reset(seed=seed, options=options)

        info["cdr_stage"] = self.stage
        info["cdr_fraction"] = self.current_fraction
        info["cdr_ranges"] = self._get_current_ranges()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Pass-through step.  Accumulates episode reward and triggers
        curriculum advancement on episode end.

        BUG FIX: episode_reward is reset to 0.0 AFTER the advancement
        check, not before.  The original reset it before appending to
        reward_history, so reward_history always received 0.0.
        Fixed ordering: accumulate → append → check → reset.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)

        if terminated or truncated:
            # 1. Record completed episode reward
            self.reward_history.append(self.episode_reward)
            self.episode_count += 1

            # 2. Check for stage advancement
            self._maybe_advance_stage()

            # 3. Periodic console log
            if self.verbose and self.episode_count % self.log_interval == 0:
                mean_r = (
                    float(np.mean(self.reward_history))
                    if self.reward_history
                    else float("nan")
                )
                threshold_str = str(
                    self.advance_thresholds.get(self.stage, "N/A (final stage)")
                )
                print(
                    f"[CDR] Episode {self.episode_count:5d} | "
                    f"Stage {self.stage} ({self.current_fraction * 100:.0f}%) | "
                    f"Rolling mean: {mean_r:8.2f} | "
                    f"Threshold: {threshold_str}"
                )

            # 4. Reset accumulator for next episode
            self.episode_reward = 0.0

        # Inject curriculum metadata into every step's info dict.
        info["cdr_stage"] = self.stage
        info["cdr_fraction"] = self.current_fraction
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Utility / research helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Return a summary dict for logging to TensorBoard or W&B.

        Usage in a training callback::

            stats = env.get_curriculum_stats()
            writer.add_scalar("curriculum/stage",       stats["stage"],       step)
            writer.add_scalar("curriculum/dr_fraction", stats["dr_fraction"], step)
        """
        mean_r = (
            float(np.mean(self.reward_history)) if self.reward_history else float("nan")
        )
        episodes_since_last_advance = self.episode_count - (
            self.stage_history[-1][0] if self.stage_history else 0
        )
        return {
            "stage": self.stage,
            "dr_fraction": self.current_fraction,
            "rolling_mean_reward": mean_r,
            "episodes_in_stage": episodes_since_last_advance,
            "total_episodes": self.episode_count,
            "stage_history": list(self.stage_history),
            "mean_reward_at_advance": list(self.mean_reward_at_advance),
            "current_ranges": self._get_current_ranges(),
        }

    def force_stage(self, stage: int):
        """
        Manually set the curriculum stage (useful for ablations / debugging).

        Example::

            env.force_stage(4)   # jump straight to full DR (= Uniform DR baseline)
            env.force_stage(0)   # drop back to nominal physics only

        BUG FIX: now raises ValueError with a clear message rather than a
        bare AssertionError, which is suppressed by pytest's assert rewriting
        and gives no useful info in production logs.
        """
        if not (0 <= stage < NUM_STAGES):
            raise ValueError(f"stage must be in [0, {NUM_STAGES - 1}], got {stage}")
        self.stage = stage
        if self.verbose:
            print(
                f"[CDR] Stage manually set to {stage} "
                f"({self.current_fraction * 100:.0f}% DR width)"
            )

    def reset_curriculum(self):
        """
        Reset all curriculum state to Stage 0.
        Useful between experiment seeds without recreating the env.
        """
        self.stage = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.reward_history.clear()
        self.stage_history.clear()
        self.mean_reward_at_advance.clear()
        if self.verbose:
            print("[CDR] Curriculum reset to Stage 0.")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience factory functions
# ══════════════════════════════════════════════════════════════════════════════


def make_cdr_env(
    env_kwargs: Optional[Dict] = None,
    **cdr_kwargs,
) -> AUVCurriculumDREnv:
    """
    Factory for use with SB3's ``make_vec_env`` / ``SubprocVecEnv``.

    Single env::

        env = make_cdr_env(env_kwargs={}, window=50, verbose=True)

    Vectorized (SB3)::

        from stable_baselines3.common.env_util import make_vec_env
        vec_env = make_vec_env(
            lambda: make_cdr_env(env_kwargs={}),
            n_envs=4,
        )
    """
    return AUVCurriculumDREnv(env_or_kwargs=env_kwargs or {}, **cdr_kwargs)


def make_uniform_dr_env(
    env_kwargs: Optional[Dict] = None,
) -> AUVCurriculumDREnv:
    """
    CDR env forced to Stage 4 = Uniform DR baseline.
    Use for the uniform DR experiment to keep identical code paths.

    Example::

        baseline_env = make_uniform_dr_env()
    """
    env = AUVCurriculumDREnv(env_or_kwargs=env_kwargs or {}, verbose=False)
    env.force_stage(4)
    return env


def make_no_dr_env(
    env_kwargs: Optional[Dict] = None,
) -> AUVCurriculumDREnv:
    """
    CDR env forced to Stage 0 = No DR (nominal physics only).
    Use for the naive SAC baseline.

    Example::

        naive_env = make_no_dr_env()
    """
    env = AUVCurriculumDREnv(env_or_kwargs=env_kwargs or {}, verbose=False)
    env.force_stage(0)
    return env


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test  (python envs/auv_cdr_env.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run:  python envs/auv_cdr_env.py
    Tests CDR logic without a real AUV env (uses CartPole as stub).
    Replace CartPole with AUVDREnv for the real integration test.
    """

    print("=" * 60)
    print("CDR Smoke Test  (CartPole stub)")
    print("=" * 60)

    # ── Stub: wrap CartPole to mimic AUVDREnv's set_dr_ranges() API ──────────
    class _CartPoleStub(gym.Wrapper):
        """Minimal AUVDREnv stand-in that accepts set_dr_ranges() calls."""

        def __init__(self):
            super().__init__(gym.make("CartPole-v1"))
            self._current_ranges: Dict[str, Tuple[float, float]] = {}

        def set_dr_ranges(self, ranges: Dict[str, Tuple[float, float]]):
            self._current_ranges = ranges  # accept without error

    stub_env = _CartPoleStub()

    cdr_env = AUVCurriculumDREnv(
        env_or_kwargs=stub_env,
        window=10,
        advance_thresholds={0: 30.0, 1: 60.0, 2: 80.0, 3: 100.0},
        verbose=True,
        log_interval=20,
    )

    # ── Simulate 200 episodes ─────────────────────────────────────────────────
    stages_seen = set()
    for ep in range(200):
        obs, info = cdr_env.reset()
        stages_seen.add(info["cdr_stage"])
        done = False
        while not done:
            action = cdr_env.action_space.sample()
            obs, reward, terminated, truncated, info = cdr_env.step(action)
            done = terminated or truncated

    # ── Verify BUG FIX: reward_history must not be all-zeros ─────────────────
    history = list(cdr_env.reward_history)
    assert any(r != 0.0 for r in history), (
        "BUG: reward_history is all zeros — episode_reward was reset "
        "before being appended (original bug)."
    )

    # ── Verify force_stage raises ValueError on bad input ────────────────────
    try:
        cdr_env.force_stage(99)
        raise AssertionError("force_stage(99) should have raised ValueError")
    except ValueError as exc:
        print(f"\n[✓] force_stage(99) raised ValueError: {exc}")

    # ── Verify reset_curriculum clears state ─────────────────────────────────
    cdr_env.reset_curriculum()
    assert cdr_env.stage == 0
    assert cdr_env.episode_count == 0
    assert len(cdr_env.reward_history) == 0
    print("[✓] reset_curriculum() cleared all state correctly.")

    # ── Final stats ───────────────────────────────────────────────────────────
    # Re-run a few episodes after reset to populate stats
    for _ in range(30):
        obs, _ = cdr_env.reset()
        done = False
        while not done:
            obs, r, te, tr, _ = cdr_env.step(cdr_env.action_space.sample())
            done = te or tr

    stats = cdr_env.get_curriculum_stats()
    print("\n── Final Curriculum Stats ──")
    for k, v in stats.items():
        if k != "current_ranges":
            print(f"  {k}: {v}")
    print("  current_ranges:")
    for param, rng in stats["current_ranges"].items():
        print(f"    {param}: [{rng[0]:.4f}, {rng[1]:.4f}]")

    # ── Check ranges are self-consistent ─────────────────────────────────────
    for param, (lo, hi) in stats["current_ranges"].items():
        assert lo <= hi, f"{param}: lo={lo} > hi={hi}"
        if param == "current_speed":
            assert lo >= 0.0, f"current_speed lo={lo} is negative"
    print("\n[✓] All range assertions passed.")

    print("\n✓ Smoke test complete — ready for Step 5: SAC training.\n")
