# Phase A — Data Collection
## Complete Step-by-Step Guide
**Do every task in order. Do not start A2 until A1 is complete.**

---

## Why Phase A Comes First

Every number in your paper comes from Phase A.
Every figure in your paper is built from Phase A data.
Every claim you make is backed by Phase A JSON files.

If you skip a step or run eval on the wrong model,
your paper numbers will be wrong.
There is no way to fix this after submission.

Do Phase A carefully. Do it once. Do it right.

---

## Before You Start Anything

### Verify your folder structure exists

Run this on Mac:

```bash
conda activate rl

# Check original runs exist
echo "=== ORIGINAL RUNS ==="
for mode in none uniform curriculum; do
    for seed in 0 1 2; do
        path=~/rl_research/auv/${mode}/${mode}_seed${seed}/best_model.zip
        if [ -f "$path" ]; then
            echo "✓ ${mode}_seed${seed}"
        else
            echo "✗ MISSING: ${mode}_seed${seed}"
        fi
    done
done

# Check master runs exist
echo ""
echo "=== MASTER RUNS ==="
for mode in master_none master_uniform master_curriculum; do
    for seed in 0 1 2; do
        path=~/rl_research/auv/${mode}/${mode}_seed${seed}/best_model.zip
        if [ -f "$path" ]; then
            echo "✓ ${mode}_seed${seed}"
        else
            echo "✗ MISSING: ${mode}_seed${seed}"
        fi
    done
done

# Check tracking runs exist
echo ""
echo "=== TRACKING RUNS ==="
for mode in tracking_none tracking_uniform tracking_curriculum; do
    path=~/rl_research/auv/${mode}/${mode}_seed0/best_model.zip
    if [ -f "$path" ]; then
        echo "✓ ${mode}_seed0"
    else
        echo "✗ MISSING: ${mode}_seed0"
    fi
done
```

Paste the output here before doing anything else.
Any MISSING entry means that training run did not complete.
You cannot run eval on a missing model.

---

## A1 — Energy Eval: Original 9 Runs

### What this is

Your original 9 runs used the goal-reaching environment
with no sensor noise in the DR parameter space.
These are the runs with seeds 0, 1, 2 for
none, uniform, and curriculum modes.

These produce your **main paper results** — Table 1.

### What you are measuring

For each of the 9 models, you run 100 episodes on the
held-out test distribution (TEST_PARAM_CONFIG) and record:

- **success_rate** — fraction of episodes where goal_dist < 0.5m
- **mean_reward** — mean total episode reward
- **std_reward** — standard deviation of episode rewards
- **mean_dist** — mean final goal distance at episode end
- **mean_energy_per_step** — mean absolute thruster command per step
- **mean_peak_thrust** — highest single thruster command seen
- **mean_episode_length** — mean steps per episode

### Theory — why held-out test distribution

During training the agent saw physics from PARAM_CONFIG ranges.
The test distribution uses wider ranges it NEVER saw during training.
This is zero-shot transfer — the agent gets no fine-tuning on test physics.

This is what makes your result meaningful.
Evaluating on training physics would just measure memorisation.
Evaluating on test physics measures generalisation.

### The test physics ranges

```python
TEST_PARAM_CONFIG = {
    "c_drag_lateral":  (0.30, 0.80),   # wider than training [0.10, 0.50]
    "c_drag_axial":    (0.12, 0.32),   # wider than training [0.04, 0.20]
    "buoyancy_offset": (-0.10, 0.15),  # wider than training [-0.05, 0.10]
    "current_speed":   (0.20, 0.60),   # wider than training [0.00, 0.30]
    "added_mass":      (0.20, 0.50),   # wider than training [0.05, 0.30]
    "act_efficiency":  (0.70, 1.00),   # wider than training [0.80, 1.00]
}
```

Every episode samples fresh physics from these wider ranges.
The agent must navigate to a goal 2-6m away under conditions
it was never directly trained on.

### How to run — on Colab (recommended)

Colab is faster for eval because it has GPU.
Each eval run takes about 5-10 minutes on T4.

**Cell 1 — Setup:**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.system("pip install -q mujoco gymnasium stable-baselines3")
os.system("git clone https://github.com/Itzlimon22/rl_robotics "
          "/content/rl_robotics 2>/dev/null || "
          "(cd /content/rl_robotics && git pull)")

import sys
for p in ['/content/rl_robotics',
          '/content/rl_robotics/envs',
          '/content/rl_robotics/scripts']:
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup complete")
```

**Cell 2 — Run eval on one model:**
```python
# Change these two lines for each model
MODE = "none"   # none / uniform / curriculum
SEED = 0        # 0 / 1 / 2

import subprocess
result = subprocess.run([
    "python", "/content/rl_robotics/scripts/eval.py",
    "--mode", MODE,
    "--seed", str(SEED),
    "--episodes", "100",
], capture_output=False)
```

**Run Cell 2 nine times, changing MODE and SEED each time:**

| Run | MODE | SEED |
|-----|------|------|
| 1 | none | 0 |
| 2 | none | 1 |
| 3 | none | 2 |
| 4 | uniform | 0 |
| 5 | uniform | 1 |
| 6 | uniform | 2 |
| 7 | curriculum | 0 |
| 8 | curriculum | 1 |
| 9 | curriculum | 2 |

### How to run — on Mac (alternative)

```bash
conda activate rl
cd ~/rl_robotics

# Run all 9 in sequence (takes ~15 minutes total)
for mode in none uniform curriculum; do
    for seed in 0 1 2; do
        echo "=== Evaluating ${mode} seed${seed} ==="
        python scripts/eval.py --mode $mode --seed $seed --episodes 100
    done
done
```

### What gets saved

After each eval run, a file is saved:

```
~/rl_research/auv/<mode>/<mode>_seed<N>/test_eval_results.json
```

Example content:
```json
{
  "mode": "curriculum",
  "seed": 1,
  "n_episodes": 100,
  "success_rate": 0.97,
  "mean_reward": 74.23,
  "std_reward": 21.60,
  "mean_dist": 0.48,
  "std_dist": 0.05,
  "mean_energy_per_step": 0.6283,
  "std_energy_per_step": 0.0612,
  "mean_peak_thrust": 0.8821,
  "mean_episode_length": 106.5
}
```

### Verification — check all 9 files exist

```bash
echo "=== CHECKING A1 COMPLETE ==="
for mode in none uniform curriculum; do
    for seed in 0 1 2; do
        path=~/rl_research/auv/${mode}/${mode}_seed${seed}/test_eval_results.json
        if [ -f "$path" ]; then
            sr=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['success_rate']*100:.1f}%\")")
            en=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['mean_energy_per_step']:.4f}\")")
            echo "✓ ${mode}_seed${seed}: success=${sr} energy=${en}"
        else
            echo "✗ MISSING: ${mode}_seed${seed}"
        fi
    done
done
```

### Expected results (based on your previous 50-episode eval)

| Mode | Seed 0 | Seed 1 | Seed 2 | Expected Mean |
|------|--------|--------|--------|---------------|
| None | ~24% | ~69% | ~89% | ~60% |
| Uniform | ~99% | ~100% | ~100% | ~100% |
| Curriculum | ~97% | ~100% | ~91% | ~96% |

Your 100-episode results may differ slightly from the
50-episode results you already have.
The 100-episode results are the ones you use in the paper.

### A1 is complete when

All 9 files exist at the paths above.
No file shows `success_rate: 0.0` (that would indicate eval failed).
Post your verification output before moving to A2.

---

## A2 — Energy Eval: Master 9 Runs

### What this is

Your master runs added sensor noise to the DR parameter space.
These runs have three additional physics parameters:
- `pos_noise_std` — Gaussian noise on position/distance observations
- `vel_noise_std` — Gaussian noise on velocity observations
- `ang_noise_std` — Gaussian noise on angular velocity observations

These represent a more realistic sim-to-real gap because
real sensors are noisy, not just physics.

### Why this is a separate result

Master runs test a harder version of the same task.
If CDR still achieves high transfer with noisy sensors,
your contribution is stronger — you are not just
robust to physics variation but also to sensor degradation.

This becomes Section 5.3 (Sensor Noise Robustness) in your paper.

### The master test distribution

Master eval uses the standard physics ranges PLUS sensor noise:

```python
MASTER_TEST_CONFIG = {
    # Standard physics ranges (same as original)
    "c_drag_lateral":  (0.30, 0.80),
    "c_drag_axial":    (0.12, 0.32),
    "buoyancy_offset": (-0.10, 0.15),
    "current_speed":   (0.20, 0.60),
    "added_mass":      (0.20, 0.50),
    "act_efficiency":  (0.70, 1.00),
    # PLUS sensor noise (not in original test)
    "pos_noise_std":   (0.05, 0.15),   # 5-15cm position noise
    "vel_noise_std":   (0.02, 0.08),   # 2-8 cm/s velocity noise
    "ang_noise_std":   (0.01, 0.04),   # 0.01-0.04 rad/s angular noise
}
```

### Important check before running

Master models have a different observation space than original models
because sensor noise is applied inside the environment.
The `vec_normalize.pkl` from a master run has different
running statistics than an original run.

You must load each master model with its own `vec_normalize.pkl`.
Never mix master and original vec_normalize files.

### How to verify master runs saved correctly

```bash
# Check master runs have the right obs_rms shape
python3 -c "
import pickle, numpy as np
from pathlib import Path

base = Path.home() / 'rl_research' / 'auv'
for mode in ['master_none', 'master_uniform', 'master_curriculum']:
    for seed in [0, 1, 2]:
        path = base / mode / f'{mode}_seed{seed}' / 'vec_normalize.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                vn = pickle.load(f)
            print(f'{mode}_seed{seed}: obs_rms shape = {vn.obs_rms.mean.shape}')
        else:
            print(f'MISSING: {mode}_seed{seed}')
"
```

The obs_rms mean shape should be `(19,)` for master runs
(18 base obs + sensor noise observable).
If it shows `(18,)` something went wrong with the master training.

### How to run master eval

The standard `eval.py` auto-detects whether a run is master
by checking if `master_` is in the mode name.
When it detects master mode it adds sensor noise to the test config.

```bash
# On Mac
conda activate rl
cd ~/rl_robotics

for mode in master_none master_uniform master_curriculum; do
    for seed in 0 1 2; do
        echo "=== Evaluating ${mode} seed${seed} ==="
        python scripts/eval.py --mode $mode --seed $seed --episodes 100
    done
done
```

Or on Colab, change MODE to `master_curriculum`, `master_uniform`, `master_none`
and run Cell 2 nine times as before.

### What gets saved

```
~/rl_research/auv/master_<mode>/master_<mode>_seed<N>/test_eval_results.json
```

The JSON includes `"eval_type": "master_test"` to distinguish
from original eval results.

### Verification

```bash
echo "=== CHECKING A2 COMPLETE ==="
for mode in master_none master_uniform master_curriculum; do
    for seed in 0 1 2; do
        path=~/rl_research/auv/${mode}/${mode}_seed${seed}/test_eval_results.json
        if [ -f "$path" ]; then
            sr=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['success_rate']*100:.1f}%\")")
            en=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['mean_energy_per_step']:.4f}\")")
            echo "✓ ${mode}_seed${seed}: success=${sr} energy=${en}"
        else
            echo "✗ MISSING: ${mode}_seed${seed}"
        fi
    done
done
```

### A2 is complete when

All 9 master eval JSON files exist.
Results are plausible — master runs may show slightly lower
success rates than original runs due to sensor noise.
If master success rates are dramatically lower than original
(e.g., original = 97%, master = 20%) something is wrong
with how sensor noise is being applied during eval.

---

## A3 — Energy Eval: Tracking Runs

### What this is

Trajectory tracking uses a lemniscate (figure-8) path.
The goal marker moves along the path at 0.3 m/s.
The agent must stay within 1.0m of the moving target.

This is a different task from goal-reaching.
It tests whether the DR training generalises
to dynamic target-following, not just static goal-reaching.

You currently only have seed 0 for all three tracking modes.
That is enough for a preliminary result but note in the paper
that multi-seed tracking evaluation is future work.

### Why tracking eval is different

Tracking uses different success criteria:
- **tracking_error** — distance from AUV to current path point
- **path_progress** — fraction of lemniscate path completed (0 to 1)
- Success = mean tracking_error < 1.0m over the full episode

You cannot use the standard eval.py for tracking.
You need to run the tracking-specific eval script.

### Tracking eval script

Create this file at `scripts/eval_tracking.py`:

```python
"""
eval_tracking.py — Evaluate trajectory tracking models
=======================================================
Loads a trained tracking model and evaluates on held-out
test physics distribution.

Reports: mean tracking error, path progress, reward, energy.

Usage:
    python scripts/eval_tracking.py --mode curriculum --seed 0
    python scripts/eval_tracking.py --mode uniform --seed 0
    python scripts/eval_tracking.py --mode none --seed 0
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT  = _SCRIPT_DIR.parent
_ENVS_DIR   = _REPO_ROOT / "envs"

for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_tracking_env import HalcyonAUVTrackingEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, TEST_PARAM_CONFIG


def resolve_paths(mode, seed):
    on_colab = os.path.exists("/content/drive/MyDrive")
    if on_colab:
        base = Path("/content/drive/MyDrive/rl_research/auv")
    else:
        base = Path.home() / "rl_research" / "auv"

    run_name = f"tracking_{mode}_seed{seed}"
    run_dir  = base / f"tracking_{mode}" / run_name

    xml_candidates = [
        _ENVS_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("/content/rl_robotics/envs/auv.xml"),
    ]
    xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        raise FileNotFoundError("auv.xml not found")

    return run_dir, xml_path


def evaluate_tracking(mode, seed, n_episodes=50):
    run_dir, xml_path = resolve_paths(mode, seed)

    model_path   = run_dir / "best_model.zip"
    vecnorm_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    print(f"\n[tracking_eval] Mode={mode} Seed={seed}")
    print(f"[tracking_eval] Loading from: {run_dir}")

    def make_env():
        # Tracking env with test physics distribution
        env = HalcyonAUVTrackingEnv(
            xml_path=str(xml_path),
            path_speed=0.3,
            tracking_threshold=1.0,
            max_tracking_error=5.0,
        )
        wrapper = AUVDomainRandomWrapper(
            env, mode=mode, seed=seed + 9999, verbose=False
        )
        wrapper.set_test_distribution()
        return wrapper

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = SAC.load(str(model_path), env=vec_env)

    print(f"[tracking_eval] Running {n_episodes} episodes on TEST distribution...")

    rewards          = []
    tracking_errors  = []
    path_progresses  = []
    energies         = []
    episode_lengths  = []

    for ep in range(n_episodes):
        obs          = vec_env.reset()
        ep_reward    = 0.0
        ep_tracking  = []
        ep_energy    = []
        ep_steps     = 0
        done         = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward   += float(reward[0])
            ep_tracking.append(info[0].get("tracking_error", 0.0))
            ep_energy.append(float(np.mean(np.abs(action[0]))))
            ep_steps    += 1

            if done[0]:
                path_progresses.append(info[0].get("path_progress", 0.0))

        rewards.append(ep_reward)
        tracking_errors.append(float(np.mean(ep_tracking)))
        energies.append(float(np.mean(ep_energy)))
        episode_lengths.append(ep_steps)

        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{n_episodes} | "
                  f"tracking_error={np.mean(tracking_errors):.3f}m | "
                  f"path_progress={np.mean(path_progresses)*100:.1f}%")

    vec_env.close()

    # Success = mean tracking error below threshold
    tracking_success = float(np.mean(
        [e < 1.0 for e in tracking_errors]
    ))

    results = {
        "mode":                  mode,
        "seed":                  seed,
        "task":                  "trajectory_tracking",
        "n_episodes":            n_episodes,
        "eval_type":             "held_out_test",
        "tracking_success_rate": tracking_success,
        "mean_tracking_error":   float(np.mean(tracking_errors)),
        "std_tracking_error":    float(np.std(tracking_errors)),
        "mean_path_progress":    float(np.mean(path_progresses)),
        "std_path_progress":     float(np.std(path_progresses)),
        "mean_reward":           float(np.mean(rewards)),
        "std_reward":            float(np.std(rewards)),
        "mean_energy_per_step":  float(np.mean(energies)),
        "std_energy_per_step":   float(np.std(energies)),
        "mean_episode_length":   float(np.mean(episode_lengths)),
    }

    print(f"\n{'='*55}")
    print(f"  TRACKING EVAL — {mode.upper()} seed={seed}")
    print(f"  Tracking success:    {results['tracking_success_rate']*100:.1f}%")
    print(f"  Mean tracking error: {results['mean_tracking_error']:.3f}m "
          f"± {results['std_tracking_error']:.3f}m")
    print(f"  Mean path progress:  {results['mean_path_progress']*100:.1f}%")
    print(f"  Mean reward:         {results['mean_reward']:.2f} "
          f"± {results['std_reward']:.2f}")
    print(f"  Energy/step:         {results['mean_energy_per_step']:.4f} "
          f"± {results['std_energy_per_step']:.4f}")
    print(f"{'='*55}")

    save_path = run_dir / "tracking_eval_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[tracking_eval] Saved → {save_path}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["none", "uniform", "curriculum"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    evaluate_tracking(args.mode, args.seed, args.episodes)


if __name__ == "__main__":
    main()
```

### How to run tracking eval

```bash
conda activate rl
cd ~/rl_robotics

# Run all three tracking seed-0 models
python scripts/eval_tracking.py --mode none       --seed 0 --episodes 50
python scripts/eval_tracking.py --mode uniform    --seed 0 --episodes 50
python scripts/eval_tracking.py --mode curriculum --seed 0 --episodes 50
```

Use 50 episodes for tracking (not 100) because tracking
episodes are longer — the agent runs for up to 500 steps
following the full lemniscate path.

### What gets saved

```
~/rl_research/auv/tracking_<mode>/tracking_<mode>_seed0/tracking_eval_results.json
```

### What good tracking results look like

| Metric | Poor | Acceptable | Good |
|--------|------|------------|------|
| Tracking success | <30% | 50-70% | >80% |
| Mean tracking error | >3.0m | 1.5-2.5m | <1.0m |
| Path progress | <30% | 50-70% | >80% |
| Mean reward | negative | 0-50 | >50 |

CDR should show lower tracking error variance than None.
That variance reduction is your tracking task finding.

### Verification

```bash
echo "=== CHECKING A3 COMPLETE ==="
for mode in none uniform curriculum; do
    path=~/rl_research/auv/tracking_${mode}/tracking_${mode}_seed0/tracking_eval_results.json
    if [ -f "$path" ]; then
        sr=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['tracking_success_rate']*100:.1f}%\")")
        te=$(python3 -c "import json; d=json.load(open('$path')); print(f\"{d['mean_tracking_error']:.3f}m\")")
        echo "✓ tracking_${mode}_seed0: success=${sr} tracking_error=${te}"
    else
        echo "✗ MISSING: tracking_${mode}_seed0"
    fi
done
```

### A3 is complete when

All three tracking eval JSON files exist.
Path progress is above 30% for at least one condition
(otherwise the models did not learn tracking at all).

---

## A4 — Extract TensorBoard Curves

### What this is

Your training runs saved metrics to TensorBoard event files.
These files contain the full learning history:
- How fast goal_dist decreased
- When success rate first crossed 80%
- How curriculum_level expanded over time
- How episode reward evolved

You need to extract this data into JSON format
so you can plot it with matplotlib for Figure 2 and Figure 3.

### Theory — what TensorBoard files contain

Each training run saved event files at:
```
~/rl_research/auv/<mode>/<mode>_seed<N>/tensorboard/<run_name>/
```

These are binary protobuf files.
The `EventAccumulator` class from the TensorBoard library
reads them and returns lists of `(step, value)` pairs.

### Install required package

```bash
conda activate rl
pip install tensorboard
```

### Create the extraction script

Save as `scripts/extract_curves.py`:

```python
"""
extract_curves.py — Extract training metrics from TensorBoard logs
==================================================================
Reads all TensorBoard event files and saves metrics to JSON.
Run this ONCE after all training is complete.

Output: ~/rl_research/auv/training_curves.json

Usage:
    python scripts/extract_curves.py
    python scripts/extract_curves.py --include-master
    python scripts/extract_curves.py --include-tracking
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator
    )
except ImportError:
    print("Install tensorboard: pip install tensorboard")
    sys.exit(1)

BASE  = Path.home() / "rl_research" / "auv"
SEEDS = [0, 1, 2]

# Metrics to extract from TensorBoard
# These keys must match what your callbacks log
METRICS = [
    "env/goal_dist",           # mean goal distance — primary metric
    "rollout/ep_rew_mean",     # mean episode reward
    "rollout/success_rate",    # training distribution success rate
    "cdr/curriculum_level",    # CDR expansion level (curriculum mode only)
    "cdr/rolling_success",     # CDR rolling success rate
    "train/actor_loss",        # actor network loss
    "train/critic_loss",       # critic network loss
    "train/ent_coef",          # entropy coefficient
    "env/auv_speed",           # agent speed
]

ORIGINAL_MODES = ["none", "uniform", "curriculum"]
MASTER_MODES   = ["master_none", "master_uniform", "master_curriculum"]
TRACKING_MODES = ["tracking_none", "tracking_uniform", "tracking_curriculum"]


def find_event_file(run_dir: Path) -> Path | None:
    """Find TensorBoard event file inside a run directory."""
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return None

    # Events may be in a subdirectory named after the run
    for subdir in tb_dir.iterdir():
        if subdir.is_dir():
            events = list(subdir.glob("events.out.tfevents.*"))
            if events:
                return subdir
        elif subdir.name.startswith("events.out.tfevents."):
            return tb_dir

    return None


def extract_metric(ea: EventAccumulator, metric: str) -> dict:
    """Extract one metric from an EventAccumulator."""
    try:
        events = ea.Scalars(metric)
        return {
            "steps":  [int(e.step)  for e in events],
            "values": [float(e.value) for e in events],
        }
    except KeyError:
        return {"steps": [], "values": []}


def extract_run(mode: str, seed: int) -> dict | None:
    """Extract all metrics from one training run."""
    run_name = f"{mode}_seed{seed}"
    run_dir  = BASE / mode / run_name
    event_dir = find_event_file(run_dir)

    if event_dir is None:
        print(f"  WARNING: No TensorBoard data for {mode}_seed{seed}")
        return None

    print(f"  Extracting {mode}_seed{seed} from {event_dir}...")

    ea = EventAccumulator(str(event_dir))
    ea.Reload()

    available = ea.Tags().get("scalars", [])

    run_data = {
        "mode": mode,
        "seed": seed,
        "available_metrics": available,
    }

    for metric in METRICS:
        run_data[metric] = extract_metric(ea, metric)
        n = len(run_data[metric]["steps"])
        if n > 0:
            print(f"    {metric}: {n} data points "
                  f"(steps {run_data[metric]['steps'][0]} "
                  f"to {run_data[metric]['steps'][-1]})")
        else:
            print(f"    {metric}: NOT FOUND in this run")

    return run_data


def extract_all(modes: list, label: str) -> dict:
    """Extract all runs for a set of modes."""
    print(f"\n=== Extracting {label} runs ===")
    output = {}

    for mode in modes:
        output[mode] = []
        for seed in SEEDS:
            run_data = extract_run(mode, seed)
            output[mode].append(run_data)

    return output


def compute_mean_std(runs: list, metric: str) -> dict:
    """
    Compute mean and std across seeds for a metric.
    Interpolates to a common step grid.
    Returns dict with steps, mean, std arrays.
    """
    # Filter out None runs and runs with no data for this metric
    valid = [r for r in runs
             if r is not None and len(r.get(metric, {}).get("steps", [])) > 0]

    if not valid:
        return {"steps": [], "mean": [], "std": []}

    # Find common step range
    max_steps = min(r[metric]["steps"][-1] for r in valid)
    n_points  = min(len(r[metric]["steps"]) for r in valid)
    step_grid = np.linspace(0, max_steps, n_points)

    interpolated = []
    for run in valid:
        steps  = np.array(run[metric]["steps"])
        values = np.array(run[metric]["values"])
        interp = np.interp(step_grid, steps, values)
        interpolated.append(interp)

    arr = np.array(interpolated)
    return {
        "steps": step_grid.tolist(),
        "mean":  np.mean(arr, axis=0).tolist(),
        "std":   np.std(arr,  axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include-master",   action="store_true")
    p.add_argument("--include-tracking", action="store_true")
    args = p.parse_args()

    all_data = {}

    # Always extract original runs
    all_data["original"] = extract_all(ORIGINAL_MODES, "original")

    if args.include_master:
        all_data["master"] = extract_all(MASTER_MODES, "master")

    if args.include_tracking:
        all_data["tracking"] = extract_all(TRACKING_MODES, "tracking")

    # Compute mean/std across seeds for plotting
    all_data["aggregated"] = {}
    for dataset_name, dataset in all_data.items():
        if dataset_name == "aggregated":
            continue
        all_data["aggregated"][dataset_name] = {}
        for mode, runs in dataset.items():
            all_data["aggregated"][dataset_name][mode] = {}
            for metric in METRICS:
                all_data["aggregated"][dataset_name][mode][metric] = \
                    compute_mean_std(runs, metric)

    # Save to JSON
    out_path = BASE / "training_curves.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f)

    print(f"\n✓ Saved to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    print("\nSummary of extracted data:")
    for dataset_name, dataset in all_data.items():
        if dataset_name == "aggregated":
            continue
        print(f"\n  {dataset_name}:")
        for mode, runs in dataset.items():
            valid_seeds = [r["seed"] for r in runs if r is not None]
            print(f"    {mode}: seeds {valid_seeds}")


if __name__ == "__main__":
    main()
```

### How to run

```bash
conda activate rl
cd ~/rl_robotics

# Extract original runs only (minimum required)
python scripts/extract_curves.py

# Extract everything including master and tracking
python scripts/extract_curves.py --include-master --include-tracking
```

### What gets saved

```
~/rl_research/auv/training_curves.json
```

This file contains all TensorBoard data structured as:

```json
{
  "original": {
    "none": [
      {
        "mode": "none",
        "seed": 0,
        "env/goal_dist": {
          "steps": [1000, 2000, 3000, ...],
          "values": [8.2, 7.9, 7.6, ...]
        },
        ...
      }
    ]
  },
  "aggregated": {
    "original": {
      "none": {
        "env/goal_dist": {
          "steps": [...],
          "mean": [...],
          "std": [...]
        }
      }
    }
  }
}
```

### Verification

```bash
python3 -c "
import json
from pathlib import Path

path = Path.home() / 'rl_research' / 'auv' / 'training_curves.json'
if not path.exists():
    print('MISSING: training_curves.json')
else:
    with open(path) as f:
        data = json.load(f)
    print(f'File size: {path.stat().st_size / 1024:.1f} KB')
    for mode in ['none', 'uniform', 'curriculum']:
        runs = data.get('original', {}).get(mode, [])
        valid = [r for r in runs if r is not None]
        print(f'{mode}: {len(valid)}/3 seeds extracted')
        for r in valid:
            n = len(r.get('env/goal_dist', {}).get('steps', []))
            print(f'  seed {r[\"seed\"]}: {n} goal_dist points')
"
```

### A4 is complete when

`training_curves.json` exists at the path above.
Each of the 9 original runs has at least 500 data points
for `env/goal_dist` (roughly one point per 2000 training steps).
The aggregated section contains mean and std arrays.

---

## A5 — CDR State Checkpoints Loaded

### What this is

During training, your CDRCheckpointCallback saved the
CDR state every 50,000 steps as JSON files:

```
~/rl_research/auv/curriculum/curriculum_seed<N>/cdr_state_50000.json
~/rl_research/auv/curriculum/curriculum_seed<N>/cdr_state_100000.json
...
~/rl_research/auv/curriculum/curriculum_seed<N>/cdr_state_1000000.json
```

Each file contains:
- `curriculum_level` — how far CDR expanded at that step
- `n_episodes` — total episodes elapsed
- `n_successes` — total successful episodes
- `cdr_ranges` — current parameter ranges at that step
- `outcome_window` — last W episode outcomes

This data becomes Figure 3 — the curriculum level progression plot.

### Why this matters for the paper

Figure 3 is your evidence that CDR actually did something different.
If curriculum_level reaches 0.8+ during training, you can show:
- The curriculum expanded appropriately as the agent improved
- The expansion correlates with increasing success rate
- Seeds that expanded more tend to transfer better

Without this data you cannot demonstrate the CDR mechanism worked.

### Check CDR checkpoints exist

```bash
echo "=== CHECKING CDR CHECKPOINTS ==="
for seed in 0 1 2; do
    echo ""
    echo "curriculum_seed${seed}:"
    ls ~/rl_research/auv/curriculum/curriculum_seed${seed}/cdr_state_*.json \
       2>/dev/null | wc -l | xargs echo "  Files found:"
    
    # Show final curriculum level
    final=~/rl_research/auv/curriculum/curriculum_seed${seed}/cdr_state.json
    if [ -f "$final" ]; then
        python3 -c "
import json
with open('$final') as f: d = json.load(f)
print(f'  Final level: {d[\"curriculum_level\"]:.3f}')
print(f'  Final episodes: {d[\"n_episodes\"]}')
print(f'  Final successes: {d[\"n_successes\"]}')
"
    else
        echo "  Final cdr_state.json MISSING"
    fi
done
```

### Extract CDR progression data

Create `scripts/extract_cdr_progression.py`:

```python
"""
extract_cdr_progression.py — Extract CDR level over training steps
==================================================================
Reads all CDR checkpoint JSON files and organises the data
for Figure 3 plotting.

Output: ~/rl_research/auv/cdr_progression.json

Usage:
    python scripts/extract_cdr_progression.py
"""

import json
import glob
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"


def extract_seed(mode_prefix: str, seed: int) -> dict | None:
    """Extract curriculum level progression for one seed."""
    run_dir = BASE / f"{mode_prefix}_curriculum" / \
              f"{mode_prefix}_curriculum_seed{seed}"

    if not run_dir.exists():
        # Try without prefix
        run_dir = BASE / "curriculum" / f"curriculum_seed{seed}"

    if not run_dir.exists():
        print(f"  WARNING: No run directory for {mode_prefix}_curriculum_seed{seed}")
        return None

    # Find all checkpoint files
    checkpoint_files = sorted(
        glob.glob(str(run_dir / "cdr_state_*.json"))
    )

    if not checkpoint_files:
        print(f"  WARNING: No CDR checkpoints in {run_dir}")
        return None

    steps          = []
    levels         = []
    n_episodes_all = []
    n_successes_all = []
    ranges_history  = []

    for cp_path in checkpoint_files:
        # Extract step number from filename
        stem = Path(cp_path).stem    # e.g. "cdr_state_50000"
        step = int(stem.split("_")[-1])

        with open(cp_path) as f:
            state = json.load(f)

        steps.append(step)
        levels.append(state.get("curriculum_level", 0.0))
        n_episodes_all.append(state.get("n_episodes", 0))
        n_successes_all.append(state.get("n_successes", 0))
        ranges_history.append(state.get("cdr_ranges", {}))

    # Compute rolling success rate at each checkpoint
    success_rates = []
    for i, (ep, suc) in enumerate(zip(n_episodes_all, n_successes_all)):
        if ep > 0:
            success_rates.append(suc / ep)
        else:
            success_rates.append(0.0)

    print(f"  {run_dir.parent.name}/seed{seed}: "
          f"{len(steps)} checkpoints, "
          f"final level={levels[-1]:.3f}")

    return {
        "seed":           seed,
        "run_dir":        str(run_dir),
        "steps":          steps,
        "curriculum_level": levels,
        "n_episodes":     n_episodes_all,
        "n_successes":    n_successes_all,
        "success_rates":  success_rates,
        "ranges_history": ranges_history,
    }


def main():
    print("=== Extracting CDR progression data ===\n")

    output = {"original": [], "master": []}

    # Original curriculum runs
    print("Original curriculum runs:")
    for seed in [0, 1, 2]:
        data = extract_seed("", seed)
        output["original"].append(data)

    # Master curriculum runs
    print("\nMaster curriculum runs:")
    for seed in [0, 1, 2]:
        data = extract_seed("master", seed)
        output["master"].append(data)

    # Save
    out_path = BASE / "cdr_progression.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved to {out_path}")

    # Summary
    print("\nSummary:")
    for dataset in ["original", "master"]:
        print(f"\n  {dataset}:")
        for data in output[dataset]:
            if data is not None:
                final_level = data["curriculum_level"][-1] if data["curriculum_level"] else 0
                n_checkpoints = len(data["steps"])
                print(f"    seed {data['seed']}: "
                      f"final_level={final_level:.3f}, "
                      f"{n_checkpoints} checkpoints")
            else:
                print(f"    seed ?: NO DATA")


if __name__ == "__main__":
    main()
```

### How to run

```bash
conda activate rl
cd ~/rl_robotics
python scripts/extract_cdr_progression.py
```

### What gets saved

```
~/rl_research/auv/cdr_progression.json
```

### Verification

```bash
python3 -c "
import json
from pathlib import Path

path = Path.home() / 'rl_research' / 'auv' / 'cdr_progression.json'
if not path.exists():
    print('MISSING: cdr_progression.json')
else:
    with open(path) as f:
        data = json.load(f)
    print('Original curriculum runs:')
    for d in data['original']:
        if d:
            final = d['curriculum_level'][-1] if d['curriculum_level'] else 0
            print(f'  seed {d[\"seed\"]}: {len(d[\"steps\"])} checkpoints, '
                  f'final level = {final:.3f}')
        else:
            print('  NO DATA')
"
```

### A5 is complete when

`cdr_progression.json` exists.
At least 2 of 3 seeds have curriculum_level > 0.5 at the end.
If all seeds show curriculum_level = 0.0 throughout,
the CDR mechanism failed during training.
In that case you still have the training data — just note
that CDR did not expand for seeds 0/1/2 and explain why
(threshold was 70% in those runs, too high).

---

## Phase A Complete — Final Verification

Run this complete check to confirm all Phase A tasks are done:

```bash
conda activate rl
cd ~/rl_robotics

python3 -c "
import json
from pathlib import Path

BASE = Path.home() / 'rl_research' / 'auv'
issues = []

# A1 — Original eval
print('A1 — Original 9 evals:')
for mode in ['none', 'uniform', 'curriculum']:
    for seed in [0, 1, 2]:
        path = BASE / mode / f'{mode}_seed{seed}' / 'test_eval_results.json'
        if path.exists():
            d = json.loads(path.read_text())
            print(f'  ✓ {mode}_seed{seed}: '
                  f'{d[\"success_rate\"]*100:.1f}% / '
                  f'energy={d[\"mean_energy_per_step\"]:.4f}')
        else:
            print(f'  ✗ MISSING: {mode}_seed{seed}')
            issues.append(f'A1 missing: {mode}_seed{seed}')

# A2 — Master eval
print()
print('A2 — Master 9 evals:')
for mode in ['master_none', 'master_uniform', 'master_curriculum']:
    for seed in [0, 1, 2]:
        path = BASE / mode / f'{mode}_seed{seed}' / 'test_eval_results.json'
        if path.exists():
            d = json.loads(path.read_text())
            print(f'  ✓ {mode}_seed{seed}: {d[\"success_rate\"]*100:.1f}%')
        else:
            print(f'  ✗ MISSING: {mode}_seed{seed}')
            issues.append(f'A2 missing: {mode}_seed{seed}')

# A3 — Tracking eval
print()
print('A3 — Tracking evals (seed 0 only):')
for mode in ['none', 'uniform', 'curriculum']:
    path = (BASE / f'tracking_{mode}' / f'tracking_{mode}_seed0' /
            'tracking_eval_results.json')
    if path.exists():
        d = json.loads(path.read_text())
        print(f'  ✓ tracking_{mode}_seed0: '
              f'error={d[\"mean_tracking_error\"]:.3f}m / '
              f'progress={d[\"mean_path_progress\"]*100:.1f}%')
    else:
        print(f'  ✗ MISSING: tracking_{mode}_seed0')
        issues.append(f'A3 missing: tracking_{mode}_seed0')

# A4 — Training curves
print()
print('A4 — Training curves:')
path = BASE / 'training_curves.json'
if path.exists():
    print(f'  ✓ training_curves.json ({path.stat().st_size//1024} KB)')
else:
    print(f'  ✗ MISSING: training_curves.json')
    issues.append('A4 missing: training_curves.json')

# A5 — CDR progression
print()
print('A5 — CDR progression:')
path = BASE / 'cdr_progression.json'
if path.exists():
    d = json.loads(path.read_text())
    for run_data in d.get('original', []):
        if run_data:
            final = run_data['curriculum_level'][-1] if run_data['curriculum_level'] else 0
            print(f'  ✓ seed {run_data[\"seed\"]}: final level = {final:.3f}')
else:
    print(f'  ✗ MISSING: cdr_progression.json')
    issues.append('A5 missing: cdr_progression.json')

# Summary
print()
if issues:
    print(f'PHASE A INCOMPLETE — {len(issues)} issues:')
    for issue in issues:
        print(f'  • {issue}')
else:
    print('PHASE A COMPLETE ✓ — All data ready for figure generation')
"
```

---

## What Comes After Phase A

Once Phase A is complete and the verification script shows
no issues, move to Phase B — Figure Generation.

Phase B uses the JSON files from Phase A as input.
Phase B produces the PDF figures that go into your paper.

The order is:
```
A1-A5 complete → B1 (pipeline diagram)
                → B2 (training curves from A4)
                → B3 (curriculum level from A5)
                → B4 (results bar from A1+A2)
                → B5 (ablation if available)
```

Do not start Phase B until all of Phase A is verified complete.

---

*Phase A guide complete.*
*Estimated time: 3-4 hours total (mostly waiting for eval to run).*
*All output files are JSON — back them up to Google Drive.*
