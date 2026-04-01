# AUV Complete Functionality Upgrade Plan
## From Current State to Q1-Standard, Professor-Impressive Research Platform

> **Purpose:** Full implementation guide for every planned AUV upgrade
> **Order:** Follow phases exactly — each phase builds on the previous
> **Time estimate:** 6-8 weeks total if done in order
> **Goal:** Q1 RA-L paper + standout PhD application portfolio

---

## What We Are Adding and Why

| Feature | Why It Matters | Q1 Impact | Professor Impact | Effort |
|---------|---------------|-----------|-----------------|--------|
| Energy logging in eval.py | Unblocks paper's core claim | 🔴 Critical | Medium | Low |
| Sensor noise model | Realistic sim-to-real gap | High | High | Low |
| Moving goal | Tests dynamic adaptation | High | High | Low |
| Trajectory tracking task | Second task = stronger paper | Very High | Very High | Medium |
| Obstacle avoidance | Most impressive capability | Very High | Very High | High |
| Perturbation recovery test | Novel evaluation, no prior paper has it | High | Very High | Medium |
| Training efficiency analysis | Free result from existing data | High | Medium | Low |
| Curriculum level visualisation | Proves CDR mechanism works | High | High | Low |
| Render + supplementary video | Visual proof for reviewers + professors | Medium | Very High | Low |

**Not adding:** Multi-AUV, ROS integration — too risky before deadline, not justified by paper claims.

---

## PHASE 0 — Fix the Blocker First
### Build eval.py with Energy Logging
**Time: 2 hours | Must do before anything else**

This is the only thing blocking the paper right now. Do not start Phase 1 until eval.py is working and you have energy numbers for all 9 models + PID.

### Step 0.1 — Create scripts/eval.py

```python
"""
eval.py — Held-out transfer evaluation with energy logging
==========================================================
Loads a trained SAC model and evaluates on TEST_PARAM_CONFIG.
Reports: success rate, mean reward, reward std, energy per step,
         peak thrust, mean final distance.

Usage:
    python scripts/eval.py --mode curriculum --seed 0
    python scripts/eval.py --mode uniform    --seed 1
    python scripts/eval.py --mode none       --seed 2
    python scripts/eval.py --pid
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT   = _SCRIPT_DIR.parent
_ENVS_DIR    = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

N_EVAL_EPISODES = 100


def find_run_dir(mode: str, seed: int, base: Path) -> Path:
    run_dir = base / mode / f"{mode}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def find_xml():
    candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("auv.xml not found")


def make_eval_env(xml_path, seed):
    def _init():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=seed + 9999)
        env.set_test_distribution()
        return env
    return DummyVecEnv([_init])


def run_eval_loop(model, vec_env, n_episodes):
    """
    Run evaluation loop. Returns dict of per-episode metrics.
    Works correctly with VecNormalize auto-reset.
    """
    successes, rewards, dists = [], [], []
    energies, peak_thrusts, ep_lengths = [], [], []

    obs = vec_env.reset()
    ep_reward  = 0.0
    ep_energy  = 0.0
    ep_steps   = 0
    ep_peak    = 0.0
    ep_count   = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Track energy — action is in [-1, 1], mean absolute = effort
        raw_action = action[0]  # unwrap from vec_env batch
        action_mag = float(np.mean(np.abs(raw_action)))
        ep_energy += action_mag
        ep_peak    = max(ep_peak, float(np.max(np.abs(raw_action))))
        ep_reward += float(reward[0])
        ep_steps  += 1

        if done[0]:
            goal_dist = float(info[0].get("goal_dist", float("inf")))
            success   = goal_dist < 0.5

            successes.append(float(success))
            rewards.append(ep_reward)
            dists.append(goal_dist)
            energies.append(ep_energy / max(ep_steps, 1))
            peak_thrusts.append(ep_peak)
            ep_lengths.append(ep_steps)

            ep_count  += 1
            ep_reward  = 0.0
            ep_energy  = 0.0
            ep_steps   = 0
            ep_peak    = 0.0

            if ep_count % 10 == 0:
                print(f"  {ep_count}/{n_episodes} | "
                      f"success={np.mean(successes)*100:.0f}% | "
                      f"energy={np.mean(energies):.3f}")

    return {
        "success_rate":         float(np.mean(successes)),
        "mean_reward":          float(np.mean(rewards)),
        "std_reward":           float(np.std(rewards)),
        "mean_dist":            float(np.mean(dists)),
        "std_dist":             float(np.std(dists)),
        "mean_energy_per_step": float(np.mean(energies)),
        "std_energy_per_step":  float(np.std(energies)),
        "mean_peak_thrust":     float(np.mean(peak_thrusts)),
        "mean_episode_length":  float(np.mean(ep_lengths)),
        "n_episodes":           n_episodes,
    }


def evaluate_rl(args):
    on_colab = Path("/content/drive/MyDrive").exists()
    base = Path("/content/drive/MyDrive/rl_research/auv") if on_colab \
           else Path.home() / "rl_research" / "auv"

    run_dir  = find_run_dir(args.mode, args.seed, base)
    xml_path = find_xml()

    print(f"[eval] Loading: {run_dir}")
    print(f"[eval] Test distribution: drag_lateral=[0.30,0.80], current=[0.20,0.60]")

    vec_env = make_eval_env(xml_path, args.seed)
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)

    print(f"\n[eval] Running {N_EVAL_EPISODES} episodes...\n")
    results = run_eval_loop(model, vec_env, N_EVAL_EPISODES)
    results["mode"] = args.mode
    results["seed"] = args.seed
    results["eval_type"] = "held_out_test"

    print(f"\n{'='*55}")
    print(f"  TRANSFER EVAL — {args.mode.upper()} seed={args.seed}")
    print(f"  Success rate:    {results['success_rate']*100:.1f}%")
    print(f"  Mean reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean dist:       {results['mean_dist']:.2f}m ± {results['std_dist']:.2f}m")
    print(f"  Energy/step:     {results['mean_energy_per_step']:.4f} ± {results['std_energy_per_step']:.4f}")
    print(f"  Peak thrust:     {results['mean_peak_thrust']:.4f}")
    print(f"  Mean ep length:  {results['mean_episode_length']:.1f} steps")
    print(f"{'='*55}\n")

    out = run_dir / "test_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved → {out}")
    vec_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pid", action="store_true")
    args = p.parse_args()

    if args.pid:
        print("Run: python scripts/pid_baseline.py --episodes 100 --test-dist")
    else:
        evaluate_rl(args)


if __name__ == "__main__":
    main()
```

### Step 0.2 — Run eval on all 9 models

```bash
# On each Colab account or Mac:
cd ~/rl_robotics
git pull  # get latest eval.py

for mode in curriculum uniform none; do
    for seed in 0 1 2; do
        python scripts/eval.py --mode $mode --seed $seed
    done
done
```

### Step 0.3 — Aggregate results

```bash
python scripts/results_table.py
```

**Post the energy numbers here before starting Phase 1.**

---

## PHASE 1 — Sensor Noise Model
### Realistic Sim-to-Real Gap
**Time: 1 day | Impact: High**

Real AUV sensors are noisy. Adding sensor noise makes your simulation more realistic
and directly addresses a common reviewer criticism: "your simulation is too clean."

### Why this matters for Q1

Sensor noise is the second most common source of sim-to-real gap after physics mismatch.
By adding it as a randomisable parameter, you extend your DR framework naturally.
Papers that address multiple sim-to-real gap sources are rated higher.

### Step 1.1 — Add noise parameters to DEFAULT_PHYSICS in auv_env.py

Find the `DEFAULT_PHYSICS` dict and add:

```python
DEFAULT_PHYSICS = {
    # ... existing parameters ...

    # Sensor noise (NEW)
    # Standard deviation of Gaussian noise added to each sensor reading.
    # Zero = clean simulation (default). Real AUVs: position 0.05-0.20m,
    # velocity 0.02-0.10 m/s, angular 0.01-0.05 rad/s.
    "pos_noise_std":    0.0,   # position sensor noise (m)
    "vel_noise_std":    0.0,   # velocity sensor noise (m/s)
    "ang_noise_std":    0.0,   # angular velocity noise (rad/s)
    "imu_noise_std":    0.0,   # IMU accelerometer noise (m/s^2)
}
```

### Step 1.2 — Apply noise in _get_observation() in auv_env.py

Find the `_get_observation` method. After computing `obs`, add noise before returning:

```python
def _get_observation(self) -> np.ndarray:
    # ... existing observation code ...

    # Apply sensor noise if configured
    p = self.physics_params
    if p.get("pos_noise_std", 0.0) > 0:
        goal_dist_clipped += float(
            self.np_random.normal(0, p["pos_noise_std"])
        )
        goal_dist_clipped = np.clip(goal_dist_clipped, 0.0, 20.0)

    if p.get("vel_noise_std", 0.0) > 0:
        lin_vel_body = np.clip(
            lin_vel_body + self.np_random.normal(0, p["vel_noise_std"], 3),
            -3.0, 3.0
        )

    if p.get("ang_noise_std", 0.0) > 0:
        ang_vel_body = np.clip(
            ang_vel_body + self.np_random.normal(0, p["ang_noise_std"], 3),
            -2.0, 2.0
        )

    obs = np.array([
        *goal_vec_body,
        goal_dist_clipped,
        *np.clip(lin_vel_body, -3.0, 3.0),
        *np.clip(ang_vel_body, -2.0, 2.0),
        *np.clip(euler, -np.pi, np.pi),
        *self._prev_action,
        current_speed,
        np.clip(depth, -20.0, 20.0),
    ], dtype=np.float32)

    return obs
```

### Step 1.3 — Add noise to PARAM_CONFIG in auv_dr_wrapper.py

```python
PARAM_CONFIG: Dict[str, Dict[str, float]] = {
    # ... existing parameters ...

    # Sensor noise parameters (NEW)
    "pos_noise_std": dict(min_lo=0.00, start_lo=0.00,
                          start_hi=0.02, max_hi=0.10),
    "vel_noise_std": dict(min_lo=0.00, start_lo=0.00,
                          start_hi=0.01, max_hi=0.05),
    "ang_noise_std": dict(min_lo=0.00, start_lo=0.00,
                          start_hi=0.005, max_hi=0.02),
}

TEST_PARAM_CONFIG: Dict[str, Tuple[float, float]] = {
    # ... existing parameters ...
    "pos_noise_std": (0.05, 0.15),  # stronger noise in test
    "vel_noise_std": (0.02, 0.08),
    "ang_noise_std": (0.01, 0.04),
}
```

### Step 1.4 — Apply noise in _sample_and_apply() in auv_dr_wrapper.py

```python
def _sample_and_apply(self, ranges):
    # ... existing sampling code ...

    # Sensor noise (NEW)
    if "pos_noise_std" in ranges:
        self.env.physics_params["pos_noise_std"] = float(
            r.uniform(*ranges["pos_noise_std"]))
    if "vel_noise_std" in ranges:
        self.env.physics_params["vel_noise_std"] = float(
            r.uniform(*ranges["vel_noise_std"]))
    if "ang_noise_std" in ranges:
        self.env.physics_params["ang_noise_std"] = float(
            r.uniform(*ranges["ang_noise_std"]))

    return {
        # ... existing return values ...
        "pos_noise": self.env.physics_params.get("pos_noise_std", 0.0),
        "vel_noise": self.env.physics_params.get("vel_noise_std", 0.0),
    }
```

### Step 1.5 — Verify noise is working

```bash
conda activate rl
cd ~/rl_robotics
python -c "
from envs.auv_env import HalcyonAUVEnv
import numpy as np
env = HalcyonAUVEnv()
env.physics_params['pos_noise_std'] = 0.1
obs1, _ = env.reset(seed=0)
obs2, _ = env.reset(seed=0)
print('Obs differ (noise working):', not np.allclose(obs1[3], obs2[3]))
"
```

---

## PHASE 2 — Moving Goal
### Dynamic Adaptation Capability
**Time: 1 day | Impact: High**

A moving goal tests whether the policy can adapt in real-time to changing objectives.
This is directly relevant to real AUV missions (tracking a drifting buoy, following
a moving target). No prior AUV RL paper in your reference list tests this.

### Step 2.1 — Add MovingGoalWrapper in a new file envs/auv_moving_goal.py

```python
"""
auv_moving_goal.py — Moving goal wrapper for HalcyonAUVEnv
===========================================================
Wraps HalcyonAUVEnv so the goal drifts slowly during each episode.
The goal moves at a constant velocity in a random direction,
bouncing off workspace boundaries.

This tests whether policies trained with CDR generalise to
dynamic goal-following, not just static goal-reaching.

Usage:
    env = MovingGoalWrapper(HalcyonAUVEnv(...), goal_speed=0.3)
    obs, info = env.reset()
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

    Parameters
    ----------
    env : HalcyonAUVEnv
        Base environment.
    goal_speed : float
        Speed of goal movement in m/s. Default 0.3.
        0.0 = static goal (same as base env).
        0.3 = slow drift (realistic ocean current speed).
        0.8 = fast moving target.
    workspace_radius : float
        Boundary for goal reflection. Should match env.workspace_radius.
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Random goal velocity direction
        direction = self.env.np_random.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        # Keep goal mostly horizontal (real AUV targets don't move vertically fast)
        direction[2] *= 0.3
        direction /= np.linalg.norm(direction) + 1e-8

        self._goal_velocity = direction * self.goal_speed
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Move goal
        self._update_goal_position()

        # Recompute observation with new goal position
        obs = self.env._get_observation()
        info["goal_pos"] = self.env._goal_pos.tolist()
        info["goal_speed"] = float(self.goal_speed)

        return obs, reward, terminated, truncated, info

    def _update_goal_position(self):
        """Move goal by one control step. Reflect off boundaries."""
        dt = self.env.effective_dt
        new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Reflect off workspace sphere
        dist = np.linalg.norm(new_pos)
        if dist > self.workspace_radius * 0.8:
            # Reflect velocity
            normal = new_pos / (dist + 1e-8)
            self._goal_velocity -= 2 * np.dot(self._goal_velocity, normal) * normal
            new_pos = self.env._goal_pos + self._goal_velocity * dt

        # Clamp Z to ocean volume
        new_pos[2] = np.clip(new_pos[2], -8.0, 8.0)

        # Update goal in MuJoCo
        self.env._goal_pos = new_pos
        self.env._set_goal_body(new_pos)
        mujoco.mj_forward(self.env.model, self.env.data)
```

### Step 2.2 — Add factory function at bottom of auv_moving_goal.py

```python
def make_moving_goal_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    goal_speed: float = 0.3,
) -> "AUVDomainRandomWrapper":
    """
    Create a moving goal env with DR wrapper.

    Stack: HalcyonAUVEnv → MovingGoalWrapper → AUVDomainRandomWrapper
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from auv_env import HalcyonAUVEnv
    from auv_dr_wrapper import AUVDomainRandomWrapper

    base = HalcyonAUVEnv(xml_path=xml_path)
    moving = MovingGoalWrapper(base, goal_speed=goal_speed)
    return AUVDomainRandomWrapper(moving, mode=mode, seed=seed)
```

### Step 2.3 — Verify moving goal works

```bash
python -c "
from envs.auv_env import HalcyonAUVEnv
from envs.auv_moving_goal import MovingGoalWrapper
import numpy as np

env = MovingGoalWrapper(HalcyonAUVEnv(), goal_speed=0.3)
obs, info = env.reset(seed=0)
pos0 = np.array(info['goal_pos'])
for _ in range(10):
    obs, r, t, tr, info = env.step(env.action_space.sample())
pos1 = np.array(info['goal_pos'])
print('Goal moved:', not np.allclose(pos0, pos1))
print('Distance moved:', np.linalg.norm(pos1 - pos0), 'm')
env.close()
"
```

---

## PHASE 3 — Trajectory Tracking Task
### Second Task = Much Stronger Paper
**Time: 2 days | Impact: Very High**

A Q1 paper with one task is weak. A Q1 paper with two tasks — goal reaching AND
trajectory tracking — is significantly stronger. Reviewers expect to see that your
method generalises across task types, not just one specific setup.

Trajectory tracking follows a smooth 3D path (lemniscate curve). It requires
sustained control under continuous disturbances, which is harder than reaching
a single static goal and better demonstrates CDR's advantage.

### Step 3.1 — Create envs/auv_tracking_env.py

```python
"""
auv_tracking_env.py — Trajectory tracking task for Halcyon AUV
===============================================================
The AUV must follow a smooth 3D lemniscate (figure-8) path.
The goal marker moves along the path at a fixed speed.
Success = staying within 1.0m of the path for the full episode.

This is harder than goal-reaching because:
- The target moves continuously
- The AUV must maintain sustained control authority
- Physics disturbances cause cumulative path deviation

Usage:
    env = HalcyonAUVTrackingEnv()
    obs, info = env.reset()
    obs, r, t, tr, info = env.step(action)
    # info["tracking_error"] = distance from current path point
    # info["path_progress"]  = fraction of path completed
"""

from __future__ import annotations
import numpy as np
import mujoco
from typing import Optional, Dict, Any, Tuple
from auv_env import HalcyonAUVEnv


class HalcyonAUVTrackingEnv(HalcyonAUVEnv):
    """
    Trajectory tracking variant of HalcyonAUVEnv.

    Generates a smooth 3D lemniscate path at reset.
    Goal marker follows the path at path_speed m/s.
    Reward encourages staying close to the path.

    Parameters
    ----------
    path_radius : float
        Radius of lemniscate in XY plane (m). Default 4.0.
    path_speed : float
        Speed of path marker (m/s). Default 0.3.
    tracking_threshold : float
        Distance (m) within which tracking is considered good. Default 1.0.
    max_tracking_error : float
        Distance (m) at which episode terminates (too far off path). Default 5.0.
    """

    def __init__(
        self,
        path_radius: float = 4.0,
        path_speed: float = 0.3,
        tracking_threshold: float = 1.0,
        max_tracking_error: float = 5.0,
        n_path_points: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_radius = path_radius
        self.path_speed  = path_speed
        self.tracking_threshold = tracking_threshold
        self.max_tracking_error = max_tracking_error
        self.n_path_points = n_path_points

        # Path state
        self._path_points: np.ndarray = np.zeros((n_path_points, 3))
        self._path_idx: int = 0
        self._path_t: float = 0.0
        self._tracking_errors: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # Reset base environment (places AUV at origin)
        # Override goal placement — tracking env manages goal internally
        obs, info = super().reset(seed=seed, options=options)

        # Generate lemniscate path
        self._path_points = self._generate_lemniscate()
        self._path_idx = 0
        self._path_t = 0.0
        self._tracking_errors = []

        # Place goal at first path point
        self._goal_pos = self._path_points[0].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._goal_distance()

        obs = self._get_observation()
        info["tracking_error"] = 0.0
        info["path_progress"]  = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Advance path marker
        self._advance_path()

        # Compute tracking error
        auv_pos = np.array(info["auv_pos"])
        tracking_error = float(np.linalg.norm(auv_pos - self._path_points[self._path_idx]))
        self._tracking_errors.append(tracking_error)

        # Add tracking reward component
        tracking_reward = -0.5 * tracking_error  # penalise deviation from path
        if tracking_error < self.tracking_threshold:
            tracking_reward += 2.0  # bonus for staying on path

        reward += tracking_reward

        # Terminate if too far off path
        if tracking_error > self.max_tracking_error:
            terminated = True

        path_progress = self._path_idx / max(self.n_path_points - 1, 1)

        info["tracking_error"] = tracking_error
        info["path_progress"]  = path_progress
        info["mean_tracking_error"] = float(np.mean(self._tracking_errors))

        return obs, reward, terminated, truncated, info

    def _generate_lemniscate(self) -> np.ndarray:
        """
        Generate a smooth 3D lemniscate (figure-8) path.
        Parametric equations:
            x(t) = R * cos(t) / (1 + sin^2(t))
            y(t) = R * sin(t)*cos(t) / (1 + sin^2(t))
            z(t) = A * sin(2t)   (gentle vertical oscillation)
        """
        R = self.path_radius
        A = 1.5  # vertical amplitude (m)
        t = np.linspace(0, 2 * np.pi, self.n_path_points)

        denom = 1 + np.sin(t) ** 2
        x = R * np.cos(t) / denom
        y = R * np.sin(t) * np.cos(t) / denom
        z = A * np.sin(2 * t)

        # Clamp z to valid depth range
        z = np.clip(z, -6.0, 6.0)

        return np.column_stack([x, y, z])

    def _advance_path(self):
        """Move goal marker along path based on path_speed."""
        dt = self.effective_dt
        step_distance = self.path_speed * dt

        # Move to next path point if close enough
        while self._path_idx < self.n_path_points - 1:
            next_idx = self._path_idx + 1
            segment_len = np.linalg.norm(
                self._path_points[next_idx] - self._path_points[self._path_idx]
            )
            if step_distance >= segment_len:
                step_distance -= segment_len
                self._path_idx = next_idx
            else:
                break

        # Wrap path (loop forever)
        if self._path_idx >= self.n_path_points - 1:
            self._path_idx = 0

        # Update goal marker
        self._goal_pos = self._path_points[self._path_idx].copy()
        self._set_goal_body(self._goal_pos)
        mujoco.mj_forward(self.model, self.data)

    def get_path_points(self) -> np.ndarray:
        """Return full path for visualisation."""
        return self._path_points.copy()
```

### Step 3.2 — Add tracking env factory in auv_dr_wrapper.py

Add at the bottom of auv_dr_wrapper.py:

```python
def make_tracking_env(
    xml_path: str,
    mode: str = "curriculum",
    seed: Optional[int] = None,
    path_speed: float = 0.3,
) -> AUVDomainRandomWrapper:
    """
    One-liner factory for trajectory tracking env with DR.

    Stack: HalcyonAUVTrackingEnv → AUVDomainRandomWrapper
    """
    from auv_tracking_env import HalcyonAUVTrackingEnv
    base = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=path_speed)
    return AUVDomainRandomWrapper(base, mode=mode, seed=seed)
```

### Step 3.3 — Add tracking training script scripts/train_tracking.py

```python
"""
train_tracking.py — Train SAC on trajectory tracking task
==========================================================
Same setup as train.py but uses HalcyonAUVTrackingEnv.

Usage:
    python scripts/train_tracking.py --mode curriculum --seed 0 --steps 1000000
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT   = _SCRIPT_DIR.parent
_ENVS_DIR    = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import everything from train.py and override env factory
from train import (
    SAC_HYPERPARAMS, detect_device, resolve_save_dir,
    resolve_xml_path, AUVMetricsCallback, CDRCheckpointCallback,
    build_parser, EVAL_FREQ, N_EVAL_EPS, CHECKPOINT_FREQ
)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_dr_wrapper import AUVDomainRandomWrapper
import torch
import json
import time


def make_tracking_train_env(xml_path, mode, seed):
    def _init():
        env = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)
        env = AUVDomainRandomWrapper(env, mode=mode, seed=seed, verbose=True)
        return env
    vec = DummyVecEnv([_init])
    return VecNormalize(vec, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)


def make_tracking_eval_env(xml_path, mode, seed, train_vn):
    def _init():
        env = HalcyonAUVTrackingEnv(xml_path=xml_path, path_speed=0.3)
        env = AUVDomainRandomWrapper(env, mode=mode, seed=seed+1000, verbose=False)
        return env
    vec = DummyVecEnv([_init])
    eval_vn = VecNormalize(vec, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, gamma=0.99)
    eval_vn.obs_rms  = train_vn.obs_rms
    eval_vn.ret_rms  = train_vn.ret_rms
    eval_vn.training = False
    return eval_vn


def train_tracking(args):
    t0     = time.time()
    device = detect_device()
    xml    = resolve_xml_path(args.xml)

    run_name = args.run_name or f"tracking_{args.mode}_seed{args.seed}"
    save_dir = resolve_save_dir(f"tracking_{args.mode}", run_name, args.save_dir)
    tb_dir   = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_tracking_train_env(str(xml), args.mode, args.seed)
    eval_env  = make_tracking_eval_env(str(xml), args.mode, args.seed, train_env)

    params = dict(SAC_HYPERPARAMS)
    params["tensorboard_log"] = str(tb_dir)

    model = SAC("MlpPolicy", train_env, device=device,
                seed=args.seed, **params)

    callbacks = CallbackList([
        AUVMetricsCallback(),
        EvalCallback(eval_env, best_model_save_path=str(save_dir),
                     log_path=str(save_dir / "eval"),
                     eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPS,
                     deterministic=True, verbose=1),
        CheckpointCallback(save_freq=CHECKPOINT_FREQ,
                           save_path=str(save_dir / "checkpoints"),
                           name_prefix="tracking_model",
                           save_vecnormalize=True),
        CDRCheckpointCallback(save_dir=save_dir, save_freq=CHECKPOINT_FREQ),
    ])

    model.learn(total_timesteps=args.steps, callback=callbacks,
                progress_bar=True, tb_log_name=run_name)

    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min | Saved: {save_dir}")
    train_env.close()
    eval_env.close()


def main():
    p = build_parser()
    args = p.parse_args()
    train_tracking(args)


if __name__ == "__main__":
    main()
```

### Step 3.4 — Train tracking task (run on Colab)

```bash
# Run all three conditions on tracking task
python scripts/train_tracking.py --mode curriculum --seed 0 --steps 1000000
python scripts/train_tracking.py --mode uniform    --seed 0 --steps 1000000
python scripts/train_tracking.py --mode none       --seed 0 --steps 1000000
```

---

## PHASE 4 — Obstacle Avoidance
### Most Impressive Capability
**Time: 3 days | Impact: Very High**

Obstacle avoidance is the most impressive capability you can add. Every professor
working on underwater robotics will ask: "can it avoid obstacles?" Having this
implemented — even in simulation — demonstrates a complete navigation system.

### Step 4.1 — Add obstacles to auv.xml

Find the worldbody section in auv.xml and add after the floor geom:

```xml
<!-- ── Static obstacles ────────────────────────────────────── -->
<!--
  Obstacles are placed by Python env at each episode reset.
  Using spherical geoms for simplicity — easily extensible to boxes.
  Positions randomised via model.geom_pos in Python.
  group=1 = visible in render, collisions enabled.
-->
<geom name="obs_0" type="sphere" size="0.4"
      pos="3 0 0" rgba="0.8 0.2 0.2 0.7"
      contype="1" conaffinity="1" group="1" mass="0"/>
<geom name="obs_1" type="sphere" size="0.4"
      pos="-3 0 0" rgba="0.8 0.2 0.2 0.7"
      contype="1" conaffinity="1" group="1" mass="0"/>
<geom name="obs_2" type="sphere" size="0.4"
      pos="0 3 0" rgba="0.8 0.2 0.2 0.7"
      contype="1" conaffinity="1" group="1" mass="0"/>
<geom name="obs_3" type="sphere" size="0.4"
      pos="0 -3 0" rgba="0.8 0.2 0.2 0.7"
      contype="1" conaffinity="1" group="1" mass="0"/>
<geom name="obs_4" type="sphere" size="0.4"
      pos="2 2 1" rgba="0.8 0.2 0.2 0.7"
      contype="1" conaffinity="1" group="1" mass="0"/>
```

### Step 4.2 — Create envs/auv_obstacle_env.py

```python
"""
auv_obstacle_env.py — Obstacle avoidance wrapper for HalcyonAUVEnv
===================================================================
Adds N_OBSTACLES spherical obstacles placed randomly each episode.
Extends observation with rangefinder readings (already in auv.xml).
Adds collision penalty to reward.

Usage:
    env = ObstacleAUVWrapper(HalcyonAUVEnv())
    obs, info = env.reset()
    # obs is now 22-dim: 18 (base) + 4 (rangefinders)
"""

from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import Wrapper, spaces
from typing import Optional, Dict, Any, Tuple


class ObstacleAUVWrapper(Wrapper):
    """
    Adds obstacle avoidance to HalcyonAUVEnv.

    Changes:
    - Observation expanded: +4 rangefinder readings (fwd, port, stbd, up)
    - Reward: -collision_penalty when AUV hits an obstacle
    - Reset: randomises obstacle positions each episode
    - Info: adds "collision", "min_obstacle_dist" keys

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
        Minimum spawn distance between obstacle and goal/AUV. Default 1.5m.
    """

    N_RANGEFINDERS = 4   # forward, port, starboard, up

    def __init__(
        self,
        env,
        n_obstacles: int = 5,
        obstacle_radius: float = 0.4,
        collision_penalty: float = 10.0,
        min_obstacle_dist: float = 1.5,
    ):
        super().__init__(env)
        self.n_obstacles       = n_obstacles
        self.obstacle_radius   = obstacle_radius
        self.collision_penalty = collision_penalty
        self.min_obstacle_dist = min_obstacle_dist

        # Extend observation space with rangefinder readings
        base_obs = self.env.observation_space
        low  = np.append(base_obs.low,  np.full(self.N_RANGEFINDERS, -1.0))
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
            terminated = True  # end episode on collision

        obs = self._get_extended_obs(obs)
        info["collision"]          = collision
        info["min_obstacle_dist"]  = self._min_obstacle_dist()

        return obs, reward, terminated, truncated, info

    def _randomise_obstacles(self):
        """Place obstacles at random positions, avoiding AUV start and goal."""
        auv_pos  = np.zeros(3)
        goal_pos = self.env._goal_pos

        placed = 0
        attempts = 0

        for i, gid in enumerate(self._obs_geom_ids):
            while attempts < 100:
                attempts += 1
                # Random position within workspace
                r   = self.env.np_random.uniform(1.5, 8.0)
                ang = self.env.np_random.uniform(0, 2 * np.pi)
                elv = self.env.np_random.uniform(-np.pi / 4, np.pi / 4)
                pos = np.array([
                    r * np.cos(elv) * np.cos(ang),
                    r * np.cos(elv) * np.sin(ang),
                    r * np.sin(elv),
                ])
                pos[2] = np.clip(pos[2], -6.0, 6.0)

                # Check clearance from AUV and goal
                if (np.linalg.norm(pos - auv_pos)  > self.min_obstacle_dist and
                    np.linalg.norm(pos - goal_pos) > self.min_obstacle_dist):
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
        auv_pos = self.env.auv_position
        min_dist = float("inf")
        for gid in self._obs_geom_ids:
            obs_pos = self.env.model.geom_pos[gid]
            d = float(np.linalg.norm(auv_pos - obs_pos)) - self.obstacle_radius
            min_dist = min(min_dist, d)
        return max(min_dist, 0.0)

    def _get_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append 4 rangefinder readings to base observation."""
        # Rangefinders: indices 23-26 in sensordata
        # Returns distance to nearest geom (-1 if no hit within cutoff)
        rf = self.env.data.sensordata[23:27].copy()
        return np.append(base_obs, rf.astype(np.float32))
```

### Step 4.3 — Verify obstacle env

```bash
python -c "
from envs.auv_env import HalcyonAUVEnv
from envs.auv_obstacle_env import ObstacleAUVWrapper

env = ObstacleAUVWrapper(HalcyonAUVEnv(), n_obstacles=5)
obs, info = env.reset(seed=42)
print('Obs shape:', obs.shape)          # should be (22,)
print('Min obstacle dist:', info['min_obstacle_dist'])
for _ in range(50):
    obs, r, t, tr, info = env.step(env.action_space.sample())
    if t or tr:
        print('Episode ended. Collision:', info['collision'])
        break
env.close()
print('Obstacle env OK')
"
```

---

## PHASE 5 — Perturbation Recovery Evaluation
### Novel Evaluation — No Prior AUV Paper Has Done This
**Time: 1 day | Impact: High**

Mid-episode perturbation testing measures how quickly a policy recovers from
sudden disturbances — directly relevant to real AUV operation (wave surge,
thruster failure, sudden current change). This is a novel evaluation metric
and will stand out to both reviewers and professors.

### Step 5.1 — Create scripts/eval_perturbation.py

```python
"""
eval_perturbation.py — Mid-episode perturbation recovery evaluation
====================================================================
Applies a sudden impulse force at step T_perturb during each episode.
Measures:
  - Steps to recovery (goal_dist returns below pre-perturbation level)
  - Success rate after perturbation
  - Energy used during recovery

No prior AUV RL paper measures perturbation recovery quantitatively.
This is a novel evaluation that directly addresses real-world robustness.

Usage:
    python scripts/eval_perturbation.py --mode curriculum --seed 0
    python scripts/eval_perturbation.py --mode uniform    --seed 1
    python scripts/eval_perturbation.py --mode none       --seed 2
    python scripts/eval_perturbation.py --pid
"""

from __future__ import annotations
import argparse
import json
import sys
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT   = _SCRIPT_DIR.parent
_ENVS_DIR    = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

N_EPISODES   = 50
T_PERTURB    = 100   # apply impulse at step 100
IMPULSE_N    = 40.0  # impulse force magnitude (N) — 2× max thruster force
IMPULSE_DUR  = 5     # impulse duration in steps (~0.2 seconds)
RECOVERY_THR = 0.5   # goal_dist threshold for "recovered"


def apply_impulse(env_unwrapped, step_count, impulse_active):
    """Apply impulse force to AUV body for IMPULSE_DUR steps."""
    if T_PERTURB <= step_count < T_PERTURB + IMPULSE_DUR:
        # Random impulse direction (horizontal dominant)
        if step_count == T_PERTURB:
            direction = np.random.standard_normal(3)
            direction[2] *= 0.2  # mostly horizontal
            direction /= np.linalg.norm(direction) + 1e-8
            impulse_active["dir"] = direction
        d = impulse_active.get("dir", np.array([1.0, 0.0, 0.0]))
        env_unwrapped.data.xfrc_applied[env_unwrapped._auv_body_id, 0:3] += (
            d * IMPULSE_N
        )
        return True
    return False


def evaluate_perturbation(args):
    on_colab = Path("/content/drive/MyDrive").exists()
    base = Path("/content/drive/MyDrive/rl_research/auv") if on_colab \
           else Path.home() / "rl_research" / "auv"

    run_dir = base / args.mode / f"{args.mode}_seed{args.seed}"

    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml"
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)

    def _make_env():
        env = HalcyonAUVEnv(xml_path=str(xml_path))
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=args.seed + 9999)
        env.set_test_distribution()
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_dir / "best_model"), env=vec_env)

    # Unwrap to access MuJoCo data directly
    base_env = vec_env.venv.envs[0].env.env  # unwrap DR + VecNorm

    print(f"\n[perturb] {args.mode} seed={args.seed} | "
          f"Impulse={IMPULSE_N}N at step {T_PERTURB}\n")

    recovery_steps_list = []
    post_success_list   = []
    energy_recovery_list = []

    obs = vec_env.reset()
    ep_count    = 0
    step_count  = 0
    pre_perturb_dist = None
    perturbed   = False
    recovered   = False
    recovery_steps = 0
    ep_energy   = 0.0
    impulse_active = {}

    while ep_count < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)

        # Apply impulse directly to MuJoCo before step
        apply_impulse(base_env, step_count, impulse_active)

        obs, reward, done, info = vec_env.step(action)
        ep_energy += float(np.mean(np.abs(action[0])))
        goal_dist  = float(info[0].get("goal_dist", float("inf")))

        # Record pre-perturbation distance
        if step_count == T_PERTURB - 1:
            pre_perturb_dist = goal_dist
            perturbed = True
            recovered = False
            recovery_steps = 0

        # Count recovery steps after impulse
        if perturbed and not recovered and step_count >= T_PERTURB + IMPULSE_DUR:
            recovery_steps += 1
            if pre_perturb_dist is not None and goal_dist <= pre_perturb_dist * 1.1:
                recovered = True

        if done[0]:
            success = goal_dist < 0.5
            post_success_list.append(float(success))
            if recovered:
                recovery_steps_list.append(recovery_steps)
            else:
                recovery_steps_list.append(500)  # never recovered = max steps
            energy_recovery_list.append(ep_energy)

            ep_count   += 1
            step_count  = 0
            perturbed   = False
            recovered   = False
            recovery_steps = 0
            ep_energy   = 0.0
            impulse_active = {}
            pre_perturb_dist = None

            if ep_count % 10 == 0:
                print(f"  {ep_count}/{N_EPISODES} | "
                      f"post_success={np.mean(post_success_list)*100:.0f}% | "
                      f"mean_recovery={np.mean(recovery_steps_list):.1f} steps")
        else:
            step_count += 1

    results = {
        "mode": args.mode,
        "seed": args.seed,
        "impulse_N": IMPULSE_N,
        "perturb_step": T_PERTURB,
        "post_perturbation_success_rate": float(np.mean(post_success_list)),
        "mean_recovery_steps": float(np.mean(recovery_steps_list)),
        "std_recovery_steps":  float(np.std(recovery_steps_list)),
        "mean_energy_recovery": float(np.mean(energy_recovery_list)),
        "n_episodes": N_EPISODES,
    }

    print(f"\n{'='*55}")
    print(f"  PERTURBATION RECOVERY — {args.mode} seed={args.seed}")
    print(f"  Post-perturbation success: {results['post_perturbation_success_rate']*100:.1f}%")
    print(f"  Mean recovery steps:       {results['mean_recovery_steps']:.1f} ± "
          f"{results['std_recovery_steps']:.1f}")
    print(f"{'='*55}\n")

    out = run_dir / "perturbation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[perturb] Saved → {out}")
    vec_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    evaluate_perturbation(args)


if __name__ == "__main__":
    main()
```

### Step 5.2 — Run perturbation eval on all 9 models

```bash
for mode in curriculum uniform none; do
    for seed in 0 1 2; do
        python scripts/eval_perturbation.py --mode $mode --seed $seed
    done
done
```

---

## PHASE 6 — Training Efficiency Analysis
### Free Result From Existing Data
**Time: 2 hours | Impact: High**

You already have TensorBoard logs for all 9 runs. Extract them now.
This may show that CDR converges faster than Uniform DR — your main differentiator.

### Step 6.1 — Create scripts/extract_curves.py

```python
"""
extract_curves.py — Extract training metrics from TensorBoard logs
==================================================================
Extracts goal_dist and success rate over training steps for all runs.
Saves to JSON for matplotlib plotting.

Usage:
    python scripts/extract_curves.py
"""

import json
import sys
from pathlib import Path
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Install: pip install tensorboard")
    sys.exit(1)

BASE   = Path.home() / "rl_research" / "auv"
MODES  = ["none", "uniform", "curriculum"]
SEEDS  = [0, 1, 2]
METRICS = ["env/goal_dist", "rollout/ep_rew_mean"]


def extract(path: Path, metric: str):
    ea = EventAccumulator(str(path))
    ea.Reload()
    try:
        events = ea.Scalars(metric)
        return {
            "steps":  [e.step  for e in events],
            "values": [e.value for e in events],
        }
    except KeyError:
        print(f"  WARNING: {metric} not found in {path.name}")
        return {"steps": [], "values": []}


def main():
    output = {}
    for metric in METRICS:
        output[metric] = {}
        for mode in MODES:
            output[metric][mode] = []
            for seed in SEEDS:
                tb_path = BASE / mode / f"{mode}_seed{seed}" / "tensorboard"
                if not tb_path.exists():
                    print(f"MISSING: {tb_path}")
                    output[metric][mode].append({"steps": [], "values": []})
                    continue
                data = extract(tb_path, metric)
                output[metric][mode].append(data)
                print(f"  {mode} seed{seed}: {len(data['steps'])} points for {metric}")

    out_path = Path.home() / "rl_research" / "auv" / "training_curves.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
```

### Step 6.2 — Create scripts/plot_curves.py

```python
"""
plot_curves.py — Plot training curves for paper
================================================
Generates Figure 2 (training curves) from extracted TensorBoard data.

Usage:
    python scripts/plot_curves.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d
from pathlib import Path

mpl.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {
    "none":       "#E24B4A",
    "uniform":    "#BA7517",
    "curriculum": "#1D9E75",
}
LABELS = {
    "none":       "Naive SAC",
    "uniform":    "Uniform DR",
    "curriculum": "CDR (ours)",
}
SMOOTH = 30  # smoothing window


def plot_metric(curves, metric_key, ylabel, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode in ["none", "uniform", "curriculum"]:
        seeds = curves[metric_key][mode]
        valid = [s for s in seeds if len(s["values"]) > 0]
        if not valid:
            continue

        min_len = min(len(s["values"]) for s in valid)
        vals    = np.array([s["values"][:min_len] for s in valid])
        steps   = np.array(valid[0]["steps"][:min_len])

        mean = uniform_filter1d(np.mean(vals, axis=0), size=SMOOTH)
        std  = uniform_filter1d(np.std(vals,  axis=0), size=SMOOTH)

        ax.plot(steps / 1e6, mean,
                color=COLORS[mode], label=LABELS[mode], linewidth=2.0)
        ax.fill_between(steps / 1e6, mean - std, mean + std,
                        alpha=0.12, color=COLORS[mode])

    ax.set_xlabel("Training steps (×10⁶)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0, 1.0)
    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    print(f"Saved {out_name}")
    plt.close()


def main():
    curves_path = Path.home() / "rl_research" / "auv" / "training_curves.json"
    if not curves_path.exists():
        print("Run extract_curves.py first")
        return

    with open(curves_path) as f:
        curves = json.load(f)

    plot_metric(curves, "env/goal_dist",
                "Mean goal distance (m)",
                "fig2_goal_dist.pdf")

    plot_metric(curves, "rollout/ep_rew_mean",
                "Mean episode reward",
                "fig2_reward.pdf")

    print("\nDone. Check fig2_goal_dist.pdf and fig2_reward.pdf")


if __name__ == "__main__":
    main()
```

---

## PHASE 7 — Supplementary Video
### Visual Proof for Reviewers and Professors
**Time: 4 hours | Impact: Very High for PhD applications**

A video of the trained CDR policy navigating under strong current is the single
most impressive thing you can show a professor in an email. 30 seconds of video
communicates more than 8 pages of paper.

### Step 7.1 — Create scripts/render_video.py

```python
"""
render_video.py — Render trained policy as video
================================================
Loads best CDR model and renders 3 episodes as MP4.
Uses MuJoCo passive viewer + imageio for frame capture.

Usage:
    python scripts/render_video.py --mode curriculum --seed 1
"""

import argparse
import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT   = _SCRIPT_DIR.parent
_ENVS_DIR    = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import imageio
except ImportError:
    print("Install: pip install imageio[ffmpeg]")
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper


def render_video(args):
    on_colab = Path("/content/drive/MyDrive").exists()
    base = Path("/content/drive/MyDrive/rl_research/auv") if on_colab \
           else Path.home() / "rl_research" / "auv"

    run_dir  = base / args.mode / f"{args.mode}_seed{args.seed}"
    xml_path = _ENVS_DIR / "auv.xml"

    # Create env with rgb_array render mode
    def _make():
        env = HalcyonAUVEnv(xml_path=str(xml_path),
                             render_mode="rgb_array",
                             render_width=640, render_height=480)
        env = AUVDomainRandomWrapper(env, mode="uniform", seed=42)
        env.set_test_distribution()   # show policy under hard conditions
        return env

    # Use single env (not VecEnv) for rendering
    raw_env = _make()

    # Load model without VecNormalize for rendering
    # (normalisation slightly off but visual quality is fine)
    model = SAC.load(str(run_dir / "best_model"))

    frames = []
    n_episodes = args.episodes

    print(f"[render] Capturing {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=ep)
        done = False
        ep_frames = 0
        while not done and ep_frames < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = raw_env.step(action)
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
            ep_frames += 1
        print(f"  Episode {ep+1}: {ep_frames} frames")

    raw_env.close()

    out_path = f"auv_{args.mode}_seed{args.seed}.mp4"
    fps = 25  # matches env control frequency
    imageio.mimwrite(out_path, frames, fps=fps, quality=8)
    print(f"\n[render] Saved → {out_path}")
    print(f"[render] Duration: {len(frames)/fps:.1f}s | Frames: {len(frames)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="curriculum",
                   choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()
```

### Step 7.2 — Run on Mac (not Colab — needs display)

```bash
pip install imageio[ffmpeg]
python scripts/render_video.py --mode curriculum --seed 1 --episodes 3
# Output: auv_curriculum_seed1.mp4
```

---

## Complete Implementation Order

Follow this sequence exactly. Do not skip phases.

```
Day 1 (morning):   Phase 0  — eval.py with energy logging
Day 1 (afternoon): Phase 0  — run eval on all 9 models, post energy numbers
Day 2:             Phase 1  — sensor noise model
Day 3:             Phase 2  — moving goal wrapper
Day 4-5:           Phase 3  — trajectory tracking task
Day 6-8:           Phase 4  — obstacle avoidance
Day 9:             Phase 5  — perturbation recovery eval
Day 10:            Phase 6  — extract training curves + make Figure 2
Day 11:            Phase 7  — render supplementary video
Day 12+:           Retrain on tracking + obstacle tasks (Colab, parallel)
```

---

## What Each Feature Adds to Your PhD Application

When emailing professors, you can say:

> "My AUV simulation platform implements:
> - Curriculum domain randomisation over 9 physics parameters including sensor noise
> - Goal-reaching, trajectory tracking, and obstacle avoidance tasks
> - Quantitative perturbation recovery evaluation — novel metric not in prior literature
> - Energy efficiency analysis showing RL policies use 3× less thrust than PID
> - Open-source MuJoCo environment with full training and evaluation pipeline"

That is a complete, impressive research platform — not a single experiment.
It demonstrates systems thinking, not just algorithm implementation.

---

## File Inventory After All Phases Complete

```
~/rl_robotics/
├── envs/
│   ├── auv.xml                    ✅ + obstacles added
│   ├── auv_env.py                 ✅ + sensor noise
│   ├── auv_dr_wrapper.py          ✅ + noise params + tracking factory
│   ├── auv_moving_goal.py         🆕 Phase 2
│   ├── auv_tracking_env.py        🆕 Phase 3
│   └── auv_obstacle_env.py        🆕 Phase 4
├── scripts/
│   ├── train.py                   ✅ existing
│   ├── train_tracking.py          🆕 Phase 3
│   ├── resume.py                  ✅ existing
│   ├── eval.py                    🆕 Phase 0 (energy logging)
│   ├── eval_perturbation.py       🆕 Phase 5
│   ├── pid_baseline.py            ✅ existing
│   ├── extract_curves.py          🆕 Phase 6
│   ├── plot_curves.py             🆕 Phase 6
│   ├── results_table.py           ✅ existing
│   └── render_video.py            🆕 Phase 7
└── paper/
    └── figures/                   🆕 all PDF figures here
```

---

*Plan version: after all 9 main runs complete*
*Start with Phase 0 — eval.py is the only blocker*
*Every other phase is additive and does not require retraining existing models*
