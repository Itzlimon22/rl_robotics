"""
pid_baseline.py — PID Controller Baseline for Halcyon X4 AUV
════════════════════════════════════════════════════════════════════════════════

Classical PID controller for 3D AUV goal-reaching.
Used as a non-learning baseline in the paper (Table 1, Figure 3).

Three independent PID loops — one per world-frame axis (X, Y, Z).
Each loop outputs a desired force; thrust allocation maps forces
to the 4-thruster X-configuration.

Usage:
    python scripts/pid_baseline.py --episodes 50
    python scripts/pid_baseline.py --episodes 50 --test-dist   # held-out ranges
    python scripts/pid_baseline.py --tune                       # interactive tuning

Paper context:
    PID is evaluated on the same held-out test distribution as SAC policies.
    Expected result: PID achieves ~30-50% success — good enough to navigate
    in calm water but degrades badly under strong current and high drag,
    which is exactly where CDR shines. This contrast is the paper's argument.

PID control law (per axis):
    e(t)     = goal_pos - auv_pos          (position error)
    u(t)     = Kp * e(t)
             + Ki * integral(e, dt)
             + Kd * (e(t) - e(t-1)) / dt   (derivative)
    u(t)     = clip(u(t), -F_max, F_max)   (thrust saturation)

Thrust allocation (world → body → thruster):
    World force F_world → body frame via rotation matrix R^T
    Body force F_body → 4 thruster commands via allocation matrix A
    A = [[1, 1, 1, 1],    # surge (all equal)
         [0, 0, 1,-1],    # sway  (left-right differential)
         [1,-1, 0, 0]]    # heave (top-bottom differential)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# PID gains (tuned manually for Halcyon X4)
# Tuning approach:
#   1. Set Ki=Kd=0, increase Kp until oscillation
#   2. Reduce Kp to 60% of oscillation value
#   3. Add Kd to damp oscillation
#   4. Add Ki to eliminate steady-state error under current
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GAINS = {
    # Proportional gain: how hard to push toward goal
    # Higher = faster response but more oscillation
    "Kp": np.array([0.8, 0.8, 0.8]),  # [x, y, z]
    # Integral gain: corrects for persistent errors (e.g. water current)
    # Too high = windup and oscillation
    "Ki": np.array([0.05, 0.05, 0.05]),
    # Derivative gain: damps oscillation, anticipates overshoot
    # Too high = noise amplification
    "Kd": np.array([0.4, 0.4, 0.4]),
    # Max force per axis (N) — matches 2-thruster differential max
    "F_max": 20.0,
    # Integral windup limit — prevents integral from growing unbounded
    "windup_limit": 5.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# PID Controller
# ─────────────────────────────────────────────────────────────────────────────


class PIDController:
    """
    Three independent PID loops for 3D position control.

    Controls AUV position in world frame.
    Outputs 4-thruster command vector ∈ [-1, 1].
    """

    def __init__(
        self,
        gains: Optional[Dict] = None,
        dt: float = 0.04,  # 25 Hz (frame_skip=4 × timestep=0.01)
    ):
        g = gains or DEFAULT_GAINS
        self.Kp = np.array(g["Kp"], dtype=np.float64)
        self.Ki = np.array(g["Ki"], dtype=np.float64)
        self.Kd = np.array(g["Kd"], dtype=np.float64)
        self.F_max = float(g["F_max"])
        self.windup_limit = float(g["windup_limit"])
        self.dt = dt

        # State
        self._integral = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)
        self._first_step = True

    def reset(self):
        """Reset integrator and derivative state. Call at episode start."""
        self._integral = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)
        self._first_step = True

    def compute(
        self,
        auv_pos_world: np.ndarray,  # [x, y, z] AUV position in world frame
        goal_pos_world: np.ndarray,  # [x, y, z] goal position in world frame
        rotation_matrix: np.ndarray,  # 3×3 body→world rotation matrix (data.xmat)
    ) -> np.ndarray:
        """
        Compute 4-thruster command from position error.

        Parameters
        ----------
        auv_pos_world   : AUV position in world frame (3,)
        goal_pos_world  : Goal position in world frame (3,)
        rotation_matrix : Body-to-world rotation matrix R (3,3)
                          data.xmat[body_id].reshape(3,3) in MuJoCo

        Returns
        -------
        ctrl : np.ndarray (4,) — thruster commands ∈ [-1, 1]
               [thrust_top, thrust_bot, thrust_left, thrust_right]
        """
        # ── Position error in world frame ─────────────────────────────────────
        error_world = goal_pos_world - auv_pos_world  # (3,)

        # ── Integral term with anti-windup ────────────────────────────────────
        self._integral += error_world * self.dt
        self._integral = np.clip(self._integral, -self.windup_limit, self.windup_limit)

        # ── Derivative term ───────────────────────────────────────────────────
        if self._first_step:
            derivative = np.zeros(3)
            self._first_step = False
        else:
            derivative = (error_world - self._prev_error) / self.dt

        self._prev_error = error_world.copy()

        # ── PID output: desired force in world frame ──────────────────────────
        F_world = (
            self.Kp * error_world + self.Ki * self._integral + self.Kd * derivative
        )

        # Clip to max force per axis
        F_world = np.clip(F_world, -self.F_max, self.F_max)

        # ── Transform world force → body frame ───────────────────────────────
        # R maps body → world, so R^T maps world → body
        R = rotation_matrix  # body → world
        F_body = R.T @ F_world  # world → body  (3,)

        # ── Thrust allocation: body force → 4 thruster commands ──────────────
        ctrl = self._allocate_thrust(F_body)

        return ctrl.astype(np.float32)

    def _allocate_thrust(self, F_body: np.ndarray) -> np.ndarray:
        """
        Map desired body-frame force to 4 thruster commands.

        Thruster layout (rear view, looking forward):
              T_top   [0]  (+Z, index 0)
          T_left [2]    T_right [3]  (±Y)
              T_bot   [1]  (-Z, index 1)

        Allocation:
          Surge  (Fx): all 4 thrusters equal        → [1, 1, 1, 1]
          Heave  (Fz): top vs bot differential       → [1,-1, 0, 0]
          Sway   (Fy): left vs right differential    → [0, 0, 1,-1]

        Each command normalised by max thrust (20N × gear).
        """
        Fx, Fy, Fz = F_body
        T_max = self.F_max  # 20N per thruster

        # Surge contribution (all thrusters)
        surge = Fx / (4 * T_max)

        # Heave contribution (top-bottom differential)
        heave = Fz / (2 * T_max)

        # Sway contribution (left-right differential)
        sway = Fy / (2 * T_max)

        # Combine per thruster: [top, bot, left, right]
        ctrl = np.array(
            [
                surge + heave,  # thrust_top:   surge + heave
                surge - heave,  # thrust_bot:   surge - heave
                surge + sway,  # thrust_left:  surge + sway
                surge - sway,  # thrust_right: surge - sway
            ]
        )

        # Clip to [-1, 1]
        return np.clip(ctrl, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────


def run_pid_eval(
    xml_path: str,
    n_episodes: int = 50,
    use_test_dist: bool = False,
    gains: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate PID controller on AUV environment.

    Parameters
    ----------
    xml_path      : Path to auv.xml
    n_episodes    : Number of evaluation episodes
    use_test_dist : If True, use held-out test distribution (paper eval)
                    If False, use training distribution
    gains         : PID gains dict. None = DEFAULT_GAINS
    seed          : Random seed
    verbose       : Print progress

    Returns
    -------
    results dict with success_rate, mean_reward, mean_dist, etc.
    """
    # Build env
    # PID needs more steps and a slightly larger goal threshold than SAC
    # goal_threshold=0.8m is fair — PID can't hover as precisely as RL
    base_env = HalcyonAUVEnv(
        xml_path=xml_path,
        max_episode_steps=1000,
        goal_threshold=0.8,
    )

    if use_test_dist:
        # Apply test distribution ranges directly
        wrapper = AUVDomainRandomWrapper(
            base_env, mode="uniform", seed=seed, verbose=False
        )
        # Override with test distribution ranges
        wrapper._test_mode = True
        if verbose:
            dist_name = "TEST (held-out)"
            print(
                f"[pid] Test ranges: "
                f"drag_lateral={TEST_PARAM_CONFIG['c_drag_lateral']}, "
                f"current={TEST_PARAM_CONFIG['current_speed']}"
            )
    else:
        wrapper = AUVDomainRandomWrapper(
            base_env, mode="uniform", seed=seed, verbose=False
        )
        dist_name = "TRAINING"

    env = wrapper
    pid = PIDController(gains=gains, dt=env.env.effective_dt)

    if verbose:
        print(f"[pid] Evaluating {n_episodes} episodes on {dist_name} distribution...")

    # Metrics
    successes = []
    rewards = []
    final_dists = []
    episode_lens = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        pid.reset()

        ep_reward = 0.0
        ep_steps = 0

        # Get initial goal position from info
        goal_pos = np.array(info["goal_pos"])

        while True:
            # Get AUV state from MuJoCo data directly
            mj_data = env.env.data
            mj_model = env.env.model
            body_id = env.env._auv_body_id

            auv_pos = mj_data.sensordata[0:3].copy()
            R = mj_data.xmat[body_id].reshape(3, 3)
            goal_pos = np.array(info.get("goal_pos", goal_pos))

            # PID control
            ctrl = pid.compute(auv_pos, goal_pos, R)

            obs, reward, terminated, truncated, info = env.step(ctrl)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                break

        # Episode metrics
        final_dist = info.get("goal_dist", float("inf"))
        success = terminated and final_dist < env.env.goal_threshold

        successes.append(float(success))
        rewards.append(ep_reward)
        final_dists.append(final_dist)
        episode_lens.append(ep_steps)

        if verbose and (ep + 1) % 10 == 0:
            sr = np.mean(successes) * 100
            print(f"  {ep + 1}/{n_episodes} episodes | success so far: {sr:.0f}%")

    env.close()

    results = {
        "controller": "PID",
        "distribution": "test" if use_test_dist else "train",
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_dist": float(np.mean(final_dists)),
        "std_dist": float(np.std(final_dists)),
        "mean_ep_len": float(np.mean(episode_lens)),
        "gains": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in (gains or DEFAULT_GAINS).items()
        },
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  PID EVAL — {dist_name} distribution")
        print(f"  Episodes:     {n_episodes}")
        print(
            f"  Success rate: {results['success_rate']:.1%}  "
            f"({int(np.sum(successes))}/{n_episodes})"
        )
        print(
            f"  Mean reward:  {results['mean_reward']:.2f} "
            f"+/- {results['std_reward']:.2f}"
        )
        print(
            f"  Mean dist:    {results['mean_dist']:.2f}m "
            f"+/- {results['std_dist']:.2f}m"
        )
        print(f"  Mean ep len:  {results['mean_ep_len']:.0f} steps")
        print(f"{'=' * 55}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gain tuning helper
# ─────────────────────────────────────────────────────────────────────────────


def tune_gains(xml_path: str, n_episodes: int = 20):
    """
    Interactive gain tuning. Tries a grid of Kp values and prints results.
    Run this once to find good gains, then hardcode them in DEFAULT_GAINS.

    Usage: python scripts/pid_baseline.py --tune
    """
    print("=== PID Gain Tuning ===")
    print("Testing Kp values with Ki=0.05, Kd=0.4 on TRAINING distribution\n")

    best_rate = 0.0
    best_gains = None

    best_inner = {}
    for kp in [8.0, 10.0, 12.0, 15.0, 20.0]:
        for kd_ratio in [0.5, 0.8, 1.2]:
            gains = dict(DEFAULT_GAINS)
            gains["Kp"] = np.array([kp, kp, kp])
            gains["Kd"] = np.array([kp * kd_ratio, kp * kd_ratio, kp * kd_ratio])
            gains["Ki"] = np.array([0.1, 0.1, 0.1])
            gains["windup_limit"] = 8.0

            results = run_pid_eval(
                xml_path,
                n_episodes=n_episodes,
                use_test_dist=False,
                gains=gains,
                seed=0,
                verbose=False,
            )
            label = f"Kp={kp:.0f} Kd={kp * kd_ratio:.1f}"
            sr = results["success_rate"]
            print(
                f"  {label} Ki=0.1 | success={sr:.1%} | "
                f"mean_dist={results['mean_dist']:.2f}m"
            )

            if sr > best_rate:
                best_rate = sr
                best_gains = gains
                best_label = label

    print(f"\nBest: {best_label} (success rate: {best_rate:.1%})")
    print(
        f"Update DEFAULT_GAINS with Kp={best_gains['Kp'][0]:.1f}, "
        f"Kd={best_gains['Kd'][0]:.1f}, Ki=0.1"
    )
    return best_gains


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate PID baseline on Halcyon AUV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--episodes", type=int, default=50, help="Number of evaluation episodes."
    )
    p.add_argument(
        "--test-dist",
        action="store_true",
        help="Use held-out test distribution (paper eval). "
        "Default: training distribution.",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run gain tuning sweep instead of evaluation.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to auv.xml. Auto-detected if not set.",
    )
    p.add_argument(
        "--save", type=str, default=None, help="Path to save results JSON. Optional."
    )
    return p


def main():
    args = build_parser().parse_args()
    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        _REPO_ROOT / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    if args.xml:
        xml_path = args.xml
    else:
        xml_path = next((str(p) for p in xml_candidates if p.exists()), None)
        if xml_path is None:
            raise FileNotFoundError("auv.xml not found. Pass --xml path/to/auv.xml")

    if args.tune:
        tune_gains(xml_path)
        return

    results = run_pid_eval(
        xml_path=xml_path,
        n_episodes=args.episodes,
        use_test_dist=args.test_dist,
        seed=args.seed,
        verbose=True,
    )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[pid] Results saved → {save_path}")


if __name__ == "__main__":
    main()
