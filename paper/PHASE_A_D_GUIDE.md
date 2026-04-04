# Phase A–D: Complete Execution Guide
## AUV Curriculum Domain Randomisation — IEEE RA-L / IROS 2026

> **How to use this file:** Every command here is copy-paste ready.
> Work top to bottom. Check boxes as you complete each task.
> All file paths are absolute. All code is complete and runnable.

---

# ═══════════════════════════════════════════════════════════
# PHASE A — DATA COLLECTION
# ═══════════════════════════════════════════════════════════

---

## A0 — Pre-flight: Verify All Models Exist

Run this first. If any model is missing, do not proceed with eval.

```bash
# On Mac terminal
conda activate rl
cd ~/rl_robotics

python3 - << 'EOF'
from pathlib import Path
import json

BASE = Path.home() / "rl_research" / "auv"

# All expected run directories
RUNS = {
    "original": [
        ("none",       0), ("none",       1), ("none",       2),
        ("uniform",    0), ("uniform",    1), ("uniform",    2),
        ("curriculum", 0), ("curriculum", 1), ("curriculum", 2),
    ],
    "master": [
        ("master_none",       0), ("master_none",       1), ("master_none",       2),
        ("master_uniform",    0), ("master_uniform",    1), ("master_uniform",    2),
        ("master_curriculum", 0), ("master_curriculum", 1), ("master_curriculum", 2),
    ],
    "tracking": [
        ("tracking_none",       0),
        ("tracking_uniform",    0),
        ("tracking_curriculum", 0),
    ],
}

print("=" * 60)
print("MODEL INVENTORY CHECK")
print("=" * 60)

total, found, missing = 0, 0, []

for group, runs in RUNS.items():
    print(f"\n── {group.upper()} ──")
    for mode, seed in runs:
        total += 1
        # Try _v1 suffix first (resumed run)
        for suffix in ["_v1", ""]:
            run_dir = BASE / mode / f"{mode}_seed{seed}{suffix}"
            best    = run_dir / "best_model.zip"
            vn      = run_dir / "vec_normalize.pkl"
            if run_dir.exists() and best.exists():
                status = "✓" if vn.exists() else "⚠ (no vec_normalize)"
                suffix_label = f" ({suffix[1:]})" if suffix else ""
                print(f"  {status} {mode}_seed{seed}{suffix_label}: {run_dir.name}")
                found += 1
                break
        else:
            print(f"  ✗ MISSING: {mode}_seed{seed}")
            missing.append(f"{mode}_seed{seed}")

print(f"\n{'='*60}")
print(f"Found:   {found}/{total}")
print(f"Missing: {len(missing)}")
if missing:
    print("Missing runs:")
    for m in missing: print(f"  - {m}")
EOF
```

---

## A1 — Energy Eval: Original 9 Runs

**What:** Evaluate the 9 original models (no sensor noise) on held-out test distribution.
**Output:** `~/rl_research/auv/<mode>/<mode>_seed<N>/energy_eval_results.json` × 9
**Time:** ~10 minutes per model on Mac (CPU), ~3 min on Colab GPU

### Option A: Run on Mac (all 9 sequentially)

```bash
conda activate rl
cd ~/rl_robotics

for mode in none uniform curriculum; do
    for seed in 0 1 2; do
        echo "════════════════════════════════════════"
        echo "Evaluating: $mode seed=$seed"
        echo "════════════════════════════════════════"
        python scripts/eval.py --mode $mode --seed $seed --episodes 100
    done
done

# Check all 9 saved
ls ~/rl_research/auv/*/*/energy_eval_results.json
```

### Option B: Run on Colab (faster, GPU)

In a new Colab cell, paste the **Standard Eval Cell** from your pipeline document,
setting `MODE` and `SEED` for each of the 9 combinations.

Or use the batch cell:

```python
# Colab batch eval cell — run all 9 original models
import subprocess, sys

REPO = "/content/rl_robotics"
subprocess.run(["bash", "-c", f"cd {REPO} && git pull"], check=True)

for mode in ["none", "uniform", "curriculum"]:
    for seed in [0, 1, 2]:
        print(f"\n{'='*50}\nEvaluating {mode} seed={seed}\n{'='*50}")
        subprocess.run([
            sys.executable, f"{REPO}/scripts/eval.py",
            "--mode", mode, "--seed", str(seed), "--episodes", "100"
        ], check=False)
```

### Verify A1 Complete

```bash
python3 - << 'EOF'
from pathlib import Path
import json

BASE  = Path.home() / "rl_research" / "auv"
MODES = ["none", "uniform", "curriculum"]
SEEDS = [0, 1, 2]

print("A1 Verification — Original Runs")
print("-" * 50)
all_ok = True
for mode in MODES:
    for seed in SEEDS:
        for suffix in ["_v1", ""]:
            p = BASE / mode / f"{mode}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f: r = json.load(f)
                sr  = r["success_rate"] * 100
                eng = r["mean_energy_per_step"]
                print(f"  ✓ {mode:12s} seed{seed}: success={sr:.1f}%  energy={eng:.4f}")
                break
        else:
            print(f"  ✗ MISSING: {mode} seed{seed}")
            all_ok = False

print()
print("A1: " + ("✅ COMPLETE" if all_ok else "❌ INCOMPLETE — run missing evals above"))
EOF
```

---

## A2 — Energy Eval: Master 9 Runs (with Sensor Noise)

**What:** Evaluate master_* models (trained with sensor noise in DR) on held-out test distribution.
**Requires:** `train_master.py` has been run for all 9 conditions.
**Output:** `~/rl_research/auv/master_<mode>/master_<mode>_seed<N>/energy_eval_results.json` × 9

### Check which master runs exist

```bash
find ~/rl_research/auv/master_* -name "best_model.zip" 2>/dev/null | sort
```

### Run master eval for each existing model

```bash
conda activate rl
cd ~/rl_robotics

for mode in master_none master_uniform master_curriculum; do
    for seed in 0 1 2; do
        # Check if model exists before running eval
        RUN_DIR="$HOME/rl_research/auv/$mode/${mode}_seed${seed}"
        if [ -f "$RUN_DIR/best_model.zip" ]; then
            echo "Evaluating $mode seed=$seed"
            python scripts/eval.py --mode $mode --seed $seed --episodes 100
        else
            echo "SKIP: $mode seed=$seed (not trained yet)"
        fi
    done
done
```

### Train missing master runs on Colab

If master runs are missing, train them first:

```python
# Colab: train master_curriculum seed=0 (example)
# This uses train_master.py which includes sensor noise in DR

import subprocess, sys
REPO = "/content/rl_robotics"
subprocess.run(["bash", "-c", f"cd {REPO} && git pull"])

MODE  = "curriculum"   # "curriculum" / "uniform" / "none"
SEED  = 0
STEPS = 1_000_000

subprocess.run([
    sys.executable, f"{REPO}/scripts/train_master.py",
    "--mode", MODE,
    "--seed", str(SEED),
    "--steps", str(STEPS),
    "--run-name", f"master_{MODE}_seed{SEED}",
])
```

> **Note on train_master.py:** This is the upgraded training script that adds
> sensor noise to `PARAM_CONFIG`. It must exist at `~/rl_robotics/scripts/train_master.py`.
> If it doesn't exist yet, copy `train.py` and add noise parameters to the DR wrapper.
> The master training produces models in `~/rl_research/auv/master_<mode>/` directories.

### Verify A2 Complete

```bash
python3 - << 'EOF'
from pathlib import Path
import json

BASE  = Path.home() / "rl_research" / "auv"
MODES = ["master_none", "master_uniform", "master_curriculum"]

print("A2 Verification — Master Runs")
print("-" * 50)
found = 0
for mode in MODES:
    for seed in range(3):
        p = BASE / mode / f"{mode}_seed{seed}" / "energy_eval_results.json"
        if p.exists():
            with open(p) as f: r = json.load(f)
            print(f"  ✓ {mode} seed{seed}: {r['success_rate']*100:.1f}%  energy={r['mean_energy_per_step']:.4f}")
            found += 1
        else:
            print(f"  ✗ MISSING: {mode} seed{seed}")

print(f"\nA2: {found}/9 complete")
EOF
```

---

## A3 — Energy Eval: Tracking Runs (seed 0)

**What:** Evaluate the 3 tracking task models (lemniscate path following).
**Output:** `~/rl_research/auv/tracking_<mode>/tracking_<mode>_seed0/energy_eval_results.json` × 3

### Run tracking eval

```bash
conda activate rl
cd ~/rl_robotics

for mode in tracking_none tracking_uniform tracking_curriculum; do
    RUN_DIR="$HOME/rl_research/auv/$mode/${mode}_seed0"
    if [ -f "$RUN_DIR/best_model.zip" ]; then
        echo "Evaluating $mode seed=0"
        python scripts/eval.py --mode $mode --seed 0 --episodes 50
    else
        echo "SKIP: $mode seed=0 not found"
    fi
done
```

### Verify A3 Complete

```bash
python3 - << 'EOF'
from pathlib import Path
import json

BASE  = Path.home() / "rl_research" / "auv"
MODES = ["tracking_none", "tracking_uniform", "tracking_curriculum"]

print("A3 Verification — Tracking Runs")
print("-" * 50)
for mode in MODES:
    p = BASE / mode / f"{mode}_seed0" / "energy_eval_results.json"
    if p.exists():
        with open(p) as f: r = json.load(f)
        print(f"  ✓ {mode}: {r['success_rate']*100:.1f}%  energy={r['mean_energy_per_step']:.4f}")
    else:
        print(f"  ✗ MISSING: {mode}")
EOF
```

---

## A4 — Extract TensorBoard Curves → training_curves.json

**What:** Pull goal_dist, episode reward, and success rate from TensorBoard logs for all 9 runs.
**Output:** `~/rl_research/auv/training_curves.json`
**Requires:** `pip install tensorboard`

### Create `scripts/extract_curves.py`

```bash
cat > ~/rl_robotics/scripts/extract_curves.py << 'PYEOF'
"""
extract_curves.py — Extract TensorBoard training metrics to JSON
================================================================
Reads TensorBoard event files from all run directories.
Saves to ~/rl_research/auv/training_curves.json for plotting.

Usage:
    python scripts/extract_curves.py
    python scripts/extract_curves.py --set master   # master_* runs
    python scripts/extract_curves.py --set all      # all runs
"""
import argparse
import json
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise ImportError("pip install tensorboard")

BASE   = Path.home() / "rl_research" / "auv"
MODES  = ["none", "uniform", "curriculum"]
M_MODES = ["master_none", "master_uniform", "master_curriculum"]
SEEDS  = [0, 1, 2]

METRICS = [
    "env/goal_dist",
    "rollout/ep_rew_mean",
    "rollout/success_rate",
    "cdr/curriculum_level",
    "cdr/rolling_success_rate",
    "cdr/c_drag_lateral",
    "cdr/current_speed",
]


def find_tb_dir(mode, seed):
    """Find TensorBoard directory — tries _v1 suffix first."""
    for suffix in ["_v1", ""]:
        run_dir = BASE / mode / f"{mode}_seed{seed}{suffix}"
        tb_dir  = run_dir / "tensorboard"
        if tb_dir.exists():
            return tb_dir
    return None


def extract_metric(tb_path, metric):
    """Extract a single metric from TensorBoard logs."""
    ea = EventAccumulator(str(tb_path))
    ea.Reload()
    try:
        events = ea.Scalars(metric)
        return {
            "steps":  [e.step  for e in events],
            "values": [e.value for e in events],
        }
    except KeyError:
        return {"steps": [], "values": []}


def extract_all(modes):
    output = {}
    for metric in METRICS:
        output[metric] = {}
        for mode in modes:
            output[metric][mode] = []
            for seed in SEEDS:
                tb_dir = find_tb_dir(mode, seed)
                if tb_dir is None:
                    print(f"  MISSING TB: {mode} seed{seed}")
                    output[metric][mode].append({"steps": [], "values": []})
                    continue
                data = extract_metric(tb_dir, metric)
                n = len(data["steps"])
                output[metric][mode].append(data)
                if n > 0:
                    print(f"  ✓ {mode} seed{seed} | {metric}: {n} points")
                else:
                    print(f"  ✗ {mode} seed{seed} | {metric}: NOT FOUND in TB")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="original",
                   choices=["original", "master", "all"])
    args = p.parse_args()

    if args.set == "original":
        modes = MODES
    elif args.set == "master":
        modes = M_MODES
    else:
        modes = MODES + M_MODES

    print(f"\nExtracting TensorBoard curves for: {modes}\n")
    output = extract_all(modes)

    out_path = BASE / "training_curves.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    print(f"\n✓ Saved → {out_path}")
    total = sum(
        len(output[m][mode][s]["steps"])
        for m in METRICS
        for mode in modes
        for s in range(len(SEEDS))
    )
    print(f"  Total data points: {total:,}")


if __name__ == "__main__":
    main()
PYEOF
```

### Run extract_curves.py

```bash
conda activate rl
cd ~/rl_robotics
python scripts/extract_curves.py

# Verify output
python3 -c "
import json
from pathlib import Path
data = json.load(open(Path.home()/'rl_research/auv/training_curves.json'))
for metric in data:
    total = sum(len(data[metric][mode][s]['steps'])
                for mode in data[metric]
                for s in range(len(data[metric][mode])))
    print(f'  {metric}: {total} total points')
print('training_curves.json ✓')
"
```

### Verify A4 Complete

```bash
python3 -c "
from pathlib import Path
p = Path.home() / 'rl_research/auv/training_curves.json'
print('A4:', '✅ COMPLETE' if p.exists() else '❌ MISSING — run extract_curves.py')
if p.exists():
    import json
    d = json.load(open(p))
    print(f'   Metrics: {list(d.keys())}')
"
```

---

## A5 — CDR State Checkpoints → Curriculum Level Data Ready

**What:** Verify CDR state JSON files exist for all curriculum runs.
These contain the curriculum_level at each 50k-step checkpoint.

### Check CDR state files

```bash
python3 - << 'EOF'
from pathlib import Path
import json

BASE = Path.home() / "rl_research" / "auv"

print("A5 Verification — CDR State Checkpoints")
print("-" * 60)

for mode_prefix in ["curriculum", "master_curriculum"]:
    for seed in range(3):
        for suffix in ["_v1", ""]:
            run_dir = BASE / mode_prefix / f"{mode_prefix}_seed{seed}{suffix}"
            if not run_dir.exists():
                continue

            cdr_files = sorted(run_dir.glob("cdr_state_*.json"))
            final_cdr = run_dir / "cdr_state.json"

            if cdr_files:
                steps = [int(f.stem.split("_")[-1]) for f in cdr_files]
                max_step = max(steps)
                with open(cdr_files[-1]) as f:
                    state = json.load(f)
                level = state.get("curriculum_level", 0)
                episodes = state.get("n_episodes", 0)
                successes = state.get("n_successes", 0)
                label = f"{mode_prefix}_seed{seed}{suffix}"
                print(f"  ✓ {label:35s} | checkpoints={len(cdr_files)} "
                      f"| max_step={max_step:,} | final_level={level:.3f} "
                      f"| episodes={episodes:,} | successes={successes:,}")
            else:
                print(f"  ✗ NO CDR checkpoints: {mode_prefix}_seed{seed}{suffix}")
            break  # found the run_dir, stop trying suffixes

print()
print("A5: CDR data is ready if all curriculum runs show ✓ above.")
EOF
```

### Generate CDR progression summary JSON

```bash
python3 - << 'EOF'
"""
Creates curriculum_progression.json — used by plot_curriculum_level.py
"""
import json
import glob
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"
output = {}

for mode_prefix in ["curriculum", "master_curriculum"]:
    output[mode_prefix] = {}
    for seed in range(3):
        for suffix in ["_v1", ""]:
            run_dir = BASE / mode_prefix / f"{mode_prefix}_seed{seed}{suffix}"
            if not run_dir.exists():
                continue
            cdr_files = sorted(run_dir.glob("cdr_state_*.json"))
            if not cdr_files:
                continue

            steps, levels, n_episodes, n_successes = [], [], [], []
            for f in cdr_files:
                step = int(f.stem.split("_")[-1])
                with open(f) as fp:
                    state = json.load(fp)
                steps.append(step)
                levels.append(state.get("curriculum_level", 0))
                n_episodes.append(state.get("n_episodes", 0))
                n_successes.append(state.get("n_successes", 0))

            output[mode_prefix][f"seed{seed}"] = {
                "steps": steps,
                "curriculum_level": levels,
                "n_episodes": n_episodes,
                "n_successes": n_successes,
            }
            print(f"  ✓ {mode_prefix} seed{seed}{suffix}: {len(steps)} checkpoints")
            break

out_path = BASE / "curriculum_progression.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → {out_path}")
EOF
```

---

# ═══════════════════════════════════════════════════════════
# PHASE B — FIGURE GENERATION
# ═══════════════════════════════════════════════════════════

**Setup — run once before any figures:**

```bash
conda activate rl
cd ~/rl_robotics
mkdir -p paper/figures paper/tables

# Install matplotlib if needed
pip install matplotlib scipy
```

---

## B1 — Figure 1: CDR Pipeline Diagram

**What:** Visual diagram of the CDR algorithm flow.
**Output:** `paper/figures/fig1_pipeline.pdf`
**Tool:** Python matplotlib (no TikZ needed)

### Create `scripts/plot_pipeline.py`

```bash
cat > ~/rl_robotics/scripts/plot_pipeline.py << 'PYEOF'
"""
plot_pipeline.py — Figure 1: CDR algorithm pipeline diagram
============================================================
Creates a clean flow diagram showing the CDR training loop.
No external tools needed — pure matplotlib.

Output: paper/figures/fig1_pipeline.pdf
Usage:  python scripts/plot_pipeline.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})

os.makedirs("paper/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# ── Color scheme ──────────────────────────────────────────────
C_BOX   = "#1D9E75"   # CDR green
C_LIGHT = "#E8F5EF"   # light green fill
C_GRAY  = "#F0F0F0"   # light gray
C_DARK  = "#2C3E50"   # dark text
C_RED   = "#E74C3C"   # contract red
C_BLUE  = "#3498DB"   # expand blue

def box(ax, x, y, w, h, text, subtext=None,
        fc=C_LIGHT, ec=C_BOX, fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.12 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            color=C_DARK, fontweight=weight, zorder=4)
    if subtext:
        ax.text(x, y - 0.22, subtext,
                ha="center", va="center", fontsize=7.5,
                color="#555555", style="italic", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555555", label=None):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color,
            lw=1.5, connectionstyle="arc3,rad=0.0"
        ), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my+0.12, label,
                ha="center", va="bottom", fontsize=7.5,
                color=color, style="italic", zorder=6)

# ── Main flow boxes ───────────────────────────────────────────
box(ax, 1.2, 2.5, 1.8, 0.7, "Episode Reset",
    "Sample φ ~ Uniform(ranges[p])", fc=C_GRAY, ec="#888888")
box(ax, 3.2, 2.5, 1.6, 0.7, "Run Episode",
    "500 steps · 25 Hz", fc=C_LIGHT, ec=C_BOX)
box(ax, 5.2, 2.5, 1.8, 0.7, "Update Window",
    "W = 50 episodes", fc=C_LIGHT, ec=C_BOX)
box(ax, 7.2, 2.5, 1.8, 0.7, "Check Success Rate",
    "sr = mean(window)", fc=C_LIGHT, ec=C_BOX)

# ── Arrows between main boxes ─────────────────────────────────
arrow(ax, 2.1, 2.5, 2.4, 2.5)
arrow(ax, 4.0, 2.5, 4.3, 2.5)
arrow(ax, 6.1, 2.5, 6.3, 2.5)

# ── EXPAND branch ─────────────────────────────────────────────
box(ax, 8.5, 3.8, 1.8, 0.6, "EXPAND",
    "sr > 0.70 → +5%", fc="#D5EEF8", ec=C_BLUE, bold=True)
arrow(ax, 8.1, 2.8, 8.5, 3.5, color=C_BLUE, label="sr > 0.70")

# ── CONTRACT branch ───────────────────────────────────────────
box(ax, 8.5, 1.2, 1.8, 0.6, "CONTRACT",
    "sr < 0.40 → -3%", fc="#FAE5E5", ec=C_RED, bold=True)
arrow(ax, 8.1, 2.2, 8.5, 1.5, color=C_RED, label="sr < 0.40")

# ── SAC update box ───────────────────────────────────────────
box(ax, 5.2, 1.0, 1.8, 0.6, "SAC Update",
    "Actor + Critic", fc="#FFF3CD", ec="#E67E22")
arrow(ax, 5.2, 2.15, 5.2, 1.3, color="#E67E22")

# ── Curriculum level tracker ─────────────────────────────────
box(ax, 3.2, 4.2, 1.8, 0.6, "Track λ ∈ [0, 1]",
    "Curriculum level", fc="#EEE", ec="#999")
ax.annotate("",
    xy=(3.2, 3.9), xytext=(5.2, 3.9),
    arrowprops=dict(arrowstyle="->", color="#999", lw=1.0,
                    connectionstyle="arc3,rad=0.0"))
ax.plot([8.5, 8.5, 3.2, 3.2], [3.5, 3.9, 3.9, 3.9],
        color="#999", lw=1.0, linestyle="--", zorder=2)

# ── Loop back arrow ───────────────────────────────────────────
ax.annotate("",
    xy=(1.2, 2.15), xytext=(1.2, 0.5),
    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.plot([1.2, 9.5, 9.5, 1.2], [0.5, 0.5, 2.5, 2.5],
        color="#555", lw=1.0, linestyle=":", zorder=2)
ax.text(5.3, 0.3, "Next episode",
        ha="center", fontsize=8, color="#555", style="italic")

# ── Title and labels ─────────────────────────────────────────
ax.set_title("Curriculum Domain Randomisation (CDR) — Training Loop",
             fontsize=11, fontweight="bold", pad=10, color=C_DARK)

# ── Legend ───────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_LIGHT, edgecolor=C_BOX,  label="CDR component"),
    mpatches.Patch(facecolor="#D5EEF8", edgecolor=C_BLUE, label="Expand ranges"),
    mpatches.Patch(facecolor="#FAE5E5", edgecolor=C_RED,  label="Contract ranges"),
    mpatches.Patch(facecolor="#FFF3CD", edgecolor="#E67E22", label="SAC update"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=8, framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
out = "paper/figures/fig1_pipeline.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"✓ Saved: {out}")
plt.close()
PYEOF
```

### Run Figure 1

```bash
conda activate rl
cd ~/rl_robotics
python scripts/plot_pipeline.py
open paper/figures/fig1_pipeline.pdf
```

---

## B2 — Figure 2: Training Curves with Shaded Std Bands

**Requires:** A4 complete (`training_curves.json` exists)
**Output:** `paper/figures/fig2_training_curves.pdf`

### Create `scripts/plot_training_curves.py`

```bash
cat > ~/rl_robotics/scripts/plot_training_curves.py << 'PYEOF'
"""
plot_training_curves.py — Figure 2: training curves with shaded std bands
=========================================================================
Requires: ~/rl_research/auv/training_curves.json (run extract_curves.py first)
Output:   paper/figures/fig2_training_curves.pdf
Usage:    python scripts/plot_training_curves.py
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})

BASE   = Path.home() / "rl_research" / "auv"
MODES  = ["none", "uniform", "curriculum"]
COLORS = {"none": "#E24B4A", "uniform": "#BA7517", "curriculum": "#1D9E75"}
LABELS = {"none": "Naive SAC", "uniform": "Uniform DR", "curriculum": "CDR (ours)"}
SMOOTH = 25

PANELS = [
    ("env/goal_dist",       "Mean goal distance (m)", None),
    ("rollout/ep_rew_mean", "Mean episode reward",    None),
]

os.makedirs("paper/figures", exist_ok=True)

curves_path = BASE / "training_curves.json"
if not curves_path.exists():
    raise FileNotFoundError(
        f"training_curves.json not found at {curves_path}\n"
        "Run: python scripts/extract_curves.py"
    )

with open(curves_path) as f:
    curves = json.load(f)

fig, axes = plt.subplots(1, len(PANELS), figsize=(5.5 * len(PANELS), 4.5))
if len(PANELS) == 1:
    axes = [axes]

for ax, (metric, ylabel, ylim) in zip(axes, PANELS):
    for mode in MODES:
        seeds = curves.get(metric, {}).get(mode, [])
        valid = [s for s in seeds if len(s.get("values", [])) > 5]
        if not valid:
            print(f"  WARNING: no data for {mode}/{metric}")
            continue

        min_len = min(len(s["values"]) for s in valid)
        if min_len < 2:
            continue

        vals  = np.array([s["values"][:min_len] for s in valid])
        steps = np.array(valid[0]["steps"][:min_len])

        mean = uniform_filter1d(np.mean(vals, axis=0), size=SMOOTH)
        std  = uniform_filter1d(np.std(vals,  axis=0), size=SMOOTH)

        ax.plot(steps / 1e6, mean,
                color=COLORS[mode], label=LABELS[mode], linewidth=2.0)
        ax.fill_between(steps / 1e6, mean - std, mean + std,
                        alpha=0.15, color=COLORS[mode])

    ax.set_xlabel("Training steps (×10⁶)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.05)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)

axes[0].legend(frameon=False, loc="upper right")

for ax, letter in zip(axes, ["(a)", "(b)"]):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)

plt.tight_layout()
out = "paper/figures/fig2_training_curves.pdf"
plt.savefig(out)
print(f"✓ Saved: {out}")
plt.close()
PYEOF
```

### Run Figure 2

```bash
conda activate rl
cd ~/rl_robotics
python scripts/plot_training_curves.py
open paper/figures/fig2_training_curves.pdf
```

---

## B3 — Figure 3: Curriculum Level Progression

**Requires:** A5 complete (`curriculum_progression.json` exists)
**Output:** `paper/figures/fig3_curriculum_level.pdf`

### Create `scripts/plot_curriculum_level.py`

```bash
cat > ~/rl_robotics/scripts/plot_curriculum_level.py << 'PYEOF'
"""
plot_curriculum_level.py — Figure 3: CDR curriculum expansion over training
===========================================================================
Output: paper/figures/fig3_curriculum_level.pdf
Usage:  python scripts/plot_curriculum_level.py
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

BASE   = Path.home() / "rl_research" / "auv"
SEED_COLORS = ["#1D9E75", "#2196F3", "#FF9800"]

os.makedirs("paper/figures", exist_ok=True)

prog_path = BASE / "curriculum_progression.json"
if not prog_path.exists():
    raise FileNotFoundError(
        f"curriculum_progression.json not found.\n"
        "Run the CDR state extraction in Phase A5."
    )

with open(prog_path) as f:
    prog = json.load(f)

# Use master_curriculum if available, else curriculum
data_key = "master_curriculum" if "master_curriculum" in prog else "curriculum"
data = prog[data_key]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for i, (seed_key, seed_data) in enumerate(sorted(data.items())):
    steps  = np.array(seed_data["steps"]) / 1e6
    levels = np.array(seed_data["curriculum_level"])
    n_eps  = np.array(seed_data["n_episodes"])
    seed_label = seed_key.replace("seed", "Seed ")

    if len(steps) == 0:
        continue

    axes[0].plot(steps, levels, color=SEED_COLORS[i],
                 label=seed_label, linewidth=2.0,
                 marker="o", markersize=4, zorder=3)

    axes[1].plot(steps, n_eps, color=SEED_COLORS[i],
                 label=seed_label, linewidth=2.0,
                 marker="o", markersize=4, zorder=3)

# Panel (a): Curriculum level
axes[0].axhline(1.0, color="#999", ls="--", lw=0.8, alpha=0.6, label="Max (λ=1)")
axes[0].axhline(0.0, color="#bbb", ls=":", lw=0.8, alpha=0.4)
axes[0].set_xlabel("Training steps (×10⁶)")
axes[0].set_ylabel("Curriculum level λ")
axes[0].set_ylim(-0.05, 1.15)
axes[0].set_xlim(0, 1.05)
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_title("(a) CDR expansion progress", loc="left",
                  fontweight="bold", fontsize=10)

# Panel (b): Number of episodes
axes[1].set_xlabel("Training steps (×10⁶)")
axes[1].set_ylabel("Cumulative episodes")
axes[1].set_xlim(0, 1.05)
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_title("(b) Training episode count", loc="left",
                  fontweight="bold", fontsize=10)

plt.suptitle(f"CDR Curriculum Progression ({data_key})",
             fontsize=11, y=1.01)
plt.tight_layout()
out = "paper/figures/fig3_curriculum_level.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"✓ Saved: {out}")
plt.close()
PYEOF
```

### Run Figure 3

```bash
conda activate rl
cd ~/rl_robotics
python scripts/plot_curriculum_level.py
open paper/figures/fig3_curriculum_level.pdf
```

---

## B4 — Figure 4: Results Bar Chart

**Requires:** A1 complete (all `energy_eval_results.json` saved)
**Output:** `paper/figures/fig4_results_bar.pdf`

### Create `scripts/plot_results.py`

```bash
cat > ~/rl_robotics/scripts/plot_results.py << 'PYEOF'
"""
plot_results.py — Figure 4: grouped bar charts for success, reward, energy
==========================================================================
Requires: energy_eval_results.json in each run folder
Output:   paper/figures/fig4_results_bar.pdf
Usage:    python scripts/plot_results.py
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

BASE    = Path.home() / "rl_research" / "auv"
MODES   = ["none", "uniform", "curriculum"]
LABELS  = ["Naive SAC", "Uniform DR", "CDR (Ours)"]
COLORS  = ["#E24B4A", "#BA7517", "#1D9E75"]
SEEDS   = [0, 1, 2]

# PID results (from your eval)
PID = {
    "success_rate": 0.03,
    "mean_reward":  -249.02,
    "mean_energy_per_step": None,   # not available from PID eval
}

PANELS = [
    ("success_rate",         "Transfer Success Rate (%)", True,  1),
    ("mean_reward",          "Mean Episode Reward",        False, 1),
    ("mean_energy_per_step", "Mean Energy per Step",       False, 4),
]

os.makedirs("paper/figures", exist_ok=True)


def load_mode(mode):
    results = []
    for seed in SEEDS:
        for suffix in ["_v1", ""]:
            p = BASE / mode / f"{mode}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f:
                    results.append(json.load(f))
                break
    return results


all_data = {m: load_mode(m) for m in MODES}

# Check data
for mode in MODES:
    n = len(all_data[mode])
    print(f"  {mode}: {n}/3 seeds loaded")

x  = np.arange(len(MODES))
bw = 0.6   # bar width

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for ax, (key, ylabel, pct, dp) in zip(axes, PANELS):
    means, stds, all_vals = [], [], []
    for mode in MODES:
        results = all_data[mode]
        if results:
            vals = [r[key] for r in results if r.get(key) is not None]
            mu = np.mean(vals) * (100 if pct else 1)
            sd = np.std(vals)  * (100 if pct else 1)
        else:
            mu, sd = 0, 0
            vals = [0]
        means.append(mu)
        stds.append(sd)
        all_vals.append([v * (100 if pct else 1) for v in vals])

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=COLORS, alpha=0.85,
                  edgecolor="white", linewidth=0.5,
                  width=bw, zorder=3)

    # Plot individual seed dots
    for xi, vals in enumerate(all_vals):
        jitter = np.random.uniform(-0.05, 0.05, len(vals))
        ax.scatter(xi + jitter, vals,
                   color="white", edgecolor=COLORS[xi],
                   s=30, zorder=5, linewidths=1.2)

    # Value labels on bars
    max_val = max(abs(m) for m in means) if means else 1
    for bar, m, s in zip(bars, means, stds):
        if abs(m) < 0.001:
            continue
        if pct:
            label = f"{m:.1f}%"
        elif dp == 4:
            label = f"{m:.4f}"
        else:
            label = f"{m:.1f}"
        offset = s + max_val * 0.04
        ypos = m + offset if m >= 0 else m - offset
        va = "bottom" if m >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                ypos, label, ha="center", va=va, fontsize=8)

    # PID reference line
    pid_val = PID.get(key)
    if pid_val is not None:
        pv = pid_val * (100 if pct else 1)
        ax.axhline(pv, color="#555", ls="--", lw=1.2, alpha=0.7,
                   label=f"PID ({pv:.1f}{'%' if pct else ''})")
        ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    idx = [k for k, _, __, ___ in PANELS].index(key)
    ax.set_title(f"({'abc'[idx]})", loc="left",
                 fontweight="bold", fontsize=10)

plt.suptitle("Transfer Evaluation — Held-Out Test Distribution",
             fontsize=12, y=1.01)
plt.tight_layout()
out = "paper/figures/fig4_results_bar.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"✓ Saved: {out}")
plt.close()
PYEOF
```

### Run Figure 4

```bash
conda activate rl
cd ~/rl_robotics
python scripts/plot_results.py
open paper/figures/fig4_results_bar.pdf
```

---

## B5 — Figure 5: Ablation Chart (if ablation runs done)

**Requires:** Ablation runs completed (5 conditions × 3 seeds = 15 runs)
**Output:** `paper/figures/fig5_ablation.pdf`

> **Note:** If ablation runs are not done, skip B5 and note in the paper:
> "Ablation experiments are left for future work."
> The paper is publishable without this figure.

### Create `scripts/plot_ablation.py`

```bash
cat > ~/rl_robotics/scripts/plot_ablation.py << 'PYEOF'
"""
plot_ablation.py — Figure 5: parameter ablation horizontal bar chart
===================================================================
Shows which physics parameter contributes most to transfer gap.
Requires ablation runs: ablation_drag, ablation_current, etc.

Output: paper/figures/fig5_ablation.pdf
Usage:  python scripts/plot_ablation.py
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

BASE  = Path.home() / "rl_research" / "auv"
os.makedirs("paper/figures", exist_ok=True)

# Expected ablation run folder names
ABLATION_RUNS = {
    "Lateral drag only":   "ablation_drag",
    "Current only":        "ablation_current",
    "Buoyancy only":       "ablation_buoyancy",
    "Added mass only":     "ablation_addedmass",
    "Act. efficiency only":"ablation_efficiency",
    "Full UDR (baseline)": "uniform",
    "No DR (baseline)":    "none",
}
SEEDS = [0, 1, 2]


def load_ablation(folder_name):
    results = []
    for seed in SEEDS:
        for suffix in ["_v1", ""]:
            p = BASE / folder_name / f"{folder_name}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f:
                    results.append(json.load(f))
                break
    return results


data = {}
for label, folder in ABLATION_RUNS.items():
    results = load_ablation(folder)
    if results:
        srs = [r["success_rate"] * 100 for r in results]
        data[label] = (np.mean(srs), np.std(srs))
        print(f"  ✓ {label}: {np.mean(srs):.1f}% ± {np.std(srs):.1f}%")
    else:
        print(f"  ✗ MISSING: {label} ({folder})")

if not data:
    print("No ablation data found. Skipping figure.")
    exit(0)

labels = list(data.keys())
means  = [data[l][0] for l in labels]
stds   = [data[l][1] for l in labels]
colors = ["#1D9E75" if "Full UDR" in l
          else "#E24B4A" if "No DR" in l
          else "#3498DB" for l in labels]

fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(labels))
bars = ax.barh(y, means, xerr=stds, capsize=4,
               color=colors, alpha=0.85,
               edgecolor="white", height=0.6)

for bar, m, s in zip(bars, means, stds):
    ax.text(m + s + 1, bar.get_y() + bar.get_height() / 2,
            f"{m:.1f}%", va="center", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Test Distribution Success Rate (%)")
ax.set_xlim(0, 115)
ax.axvline(x=means[-2] if len(means) >= 2 else 100,
           color="#1D9E75", ls="--", lw=1.0, alpha=0.5, label="Full UDR")
ax.grid(axis="x", alpha=0.3)
ax.set_title("(a) Parameter Ablation — Transfer Success",
             loc="left", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
out = "paper/figures/fig5_ablation.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"✓ Saved: {out}")
plt.close()
PYEOF
```

---

## Run All Figures at Once

```bash
conda activate rl
cd ~/rl_robotics

echo "=== Generating all figures ==="
python scripts/plot_pipeline.py
python scripts/plot_training_curves.py
python scripts/plot_curriculum_level.py
python scripts/plot_results.py
python scripts/plot_ablation.py 2>/dev/null || echo "(ablation skipped — no data)"

echo ""
echo "=== Generated figures ==="
ls -la paper/figures/
```

---

# ═══════════════════════════════════════════════════════════
# PHASE C — PAPER WRITING
# ═══════════════════════════════════════════════════════════

---

## C0 — LaTeX Setup

```bash
# Create paper directory structure
mkdir -p ~/rl_robotics/paper/sections
mkdir -p ~/rl_robotics/paper/figures
mkdir -p ~/rl_robotics/paper/tables

# Download IEEE RA-L template
cd ~/rl_robotics/paper
curl -O https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/Conference-LaTeX-template_7-9-18.zip
unzip Conference-LaTeX-template_7-9-18.zip -d template/
cp template/*.cls .
```

**File to create:** `paper/main.tex`

```latex
% paper/main.tex
\documentclass[letterpaper, 10pt, conference]{ieeeconf}
\usepackage{amsmath,amssymb,graphicx,algorithm,algpseudocode}
\usepackage{booktabs,multirow,xcolor,hyperref,url,subcaption}

\newcommand{\CDR}{\textsc{CDR}}
\newcommand{\UDR}{\textsc{UDR}}

\title{\LARGE\bf Curriculum Domain Randomisation for Robust and\\
Energy-Efficient AUV Control}

\author{[Author -- remove for double-anonymous]}

\begin{document}
\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/experiments}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

---

## C1 — Read Priority-1 Papers

**Time: ~5 hours total. Do this before writing any section.**

```bash
# Create reading notes directory
mkdir -p ~/rl_robotics/paper/reading_notes

# Papers to download and read (in order):
# 1. arxiv.org/abs/1801.01290  — SAC (Haarnoja 2018) — 1 hour
# 2. arxiv.org/abs/1703.06907  — Domain Randomisation (Tobin 2017) — 30 min
# 3. arxiv.org/abs/1910.07113  — ADR/Rubik's Cube (Akkaya 2019) — 1.5 hours
# 4. Fossen 2011 Chapter 6     — AUV hydrodynamics — 1 hour
# 5. arxiv.org/abs/2410.00120  — Learning to Swim (Arndt 2024) — 1 hour

# Create a reading note for each paper:
touch ~/rl_robotics/paper/reading_notes/haarnoja2018.md
touch ~/rl_robotics/paper/reading_notes/tobin2017.md
touch ~/rl_robotics/paper/reading_notes/akkaya2019.md
touch ~/rl_robotics/paper/reading_notes/fossen2011.md
touch ~/rl_robotics/paper/reading_notes/arndt2024.md
```

**For each paper, fill in `paper/reading_notes/<key>.md`:**

```markdown
## [Paper Title]
- **Date read:** [date]
- **One-line summary:** [what they did]
- **Their key result:** [their main number]
- **My differentiator:** [exactly how my work differs]
- **Cite in:** [which section]
- **Key equation:** [any equation I will use]
```

---

## C2 — Section 4: Experimental Setup

**File:** `~/rl_robotics/paper/sections/experiments.tex`

```bash
# Generate results table for Section 4
conda activate rl
cd ~/rl_robotics
python scripts/results_table.py --set original --latex --stats \
    > paper/tables/table4_main_results.tex

# Check it compiled
cat paper/tables/table4_main_results.tex
```

Create `paper/sections/experiments.tex` — write after all eval data confirmed.
Use numbers from `energy_eval_results.json` files.

**Key numbers to fill in:**

```bash
python3 - << 'EOF'
import json
from pathlib import Path
import numpy as np

BASE  = Path.home() / "rl_research" / "auv"
MODES = ["none", "uniform", "curriculum"]

print("Numbers for Section 4 / Table 4:")
print("-" * 60)
for mode in MODES:
    results = []
    for seed in range(3):
        for suffix in ["_v1", ""]:
            p = BASE / mode / f"{mode}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f: results.append(json.load(f))
                break
    if results:
        sr  = [r["success_rate"]*100 for r in results]
        rew = [r["mean_reward"]      for r in results]
        eng = [r["mean_energy_per_step"] for r in results]
        print(f"{mode:12s}: success={np.mean(sr):.1f}±{np.std(sr):.1f}%  "
              f"reward={np.mean(rew):.2f}±{np.std(rew):.2f}  "
              f"energy={np.mean(eng):.4f}±{np.std(eng):.4f}")
EOF
```

---

## C3 — Section 5: Results

**File:** `~/rl_robotics/paper/sections/results.tex`

Key numbers to include (run before writing):

```bash
python3 - << 'EOF'
import json, numpy as np
from pathlib import Path
from scipy import stats

BASE  = Path.home() / "rl_research" / "auv"

def load(mode):
    r = []
    for seed in range(3):
        for suffix in ["_v1", ""]:
            p = BASE / mode / f"{mode}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f: r.append(json.load(f))
                break
    return r

cdr = load("curriculum")
udr = load("uniform")
naive = load("none")

if cdr and udr:
    cdr_eng  = [r["mean_energy_per_step"] for r in cdr]
    udr_eng  = [r["mean_energy_per_step"] for r in udr]
    diff_pct = (np.mean(udr_eng) - np.mean(cdr_eng)) / np.mean(udr_eng) * 100
    t, p = stats.ttest_ind(cdr_eng, udr_eng, equal_var=False)
    d = (np.mean(udr_eng) - np.mean(cdr_eng)) / np.sqrt(
        (np.std(udr_eng)**2 + np.std(cdr_eng)**2) / 2)
    print(f"Energy saving CDR vs UDR: {diff_pct:.1f}%  p={p:.3f}  Cohen's d={d:.2f}")
    print(f"  CDR energy:  {np.mean(cdr_eng):.4f} ± {np.std(cdr_eng):.4f}")
    print(f"  UDR energy:  {np.mean(udr_eng):.4f} ± {np.std(udr_eng):.4f}")

if naive:
    naive_sr = [r["success_rate"]*100 for r in naive]
    print(f"\nNaive SAC success: {np.mean(naive_sr):.1f} ± {np.std(naive_sr):.1f}%")

print("\nPID: 100% training → 3% test (hard-coded from your eval)")
EOF
```

---

## C4 — Section 6: Discussion

**File:** `~/rl_robotics/paper/sections/discussion.tex`

Write after C3. Reference the energy numbers and statistical tests.
See the discussion template in your pipeline document.

---

## C5 — Section 2: Related Work

**File:** `~/rl_robotics/paper/sections/related_work.tex`

Write after reading all Priority-1 and Priority-2 papers (C1).
See the related work template in your pipeline document.

---

## C6 — Section 1: Introduction

**File:** `~/rl_robotics/paper/sections/introduction.tex`

Write second-to-last. Uses the PID 100%→3% hook as the opening.

---

## C7 — Section 7: Conclusion

**File:** `~/rl_robotics/paper/sections/conclusion.tex`

Write after all other sections. One paragraph, three findings.

---

## C8 — Abstract

**File:** `~/rl_robotics/paper/sections/abstract.tex`

Write very last. 150–200 words. Fill in [X] placeholders with actual numbers.

---

# ═══════════════════════════════════════════════════════════
# PHASE D — SUBMISSION
# ═══════════════════════════════════════════════════════════

---

## D1 — Full LaTeX Compile, No Errors

```bash
cd ~/rl_robotics/paper

# Install LaTeX (Mac)
# brew install --cask mactex   # if not installed

# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Check for errors
grep -c "Error" main.log
grep -c "Warning" main.log

# Check page count
pdfinfo main.pdf | grep Pages

# Open PDF
open main.pdf
```

**Page limit check:**

```bash
# RA-L: 8 pages maximum (including references)
# IROS: 6 pages + 2 reference pages = 8 total
python3 -c "
import subprocess
result = subprocess.run(['pdfinfo', 'paper/main.pdf'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Pages' in line:
        pages = int(line.split(':')[1].strip())
        print(f'Pages: {pages}/8 ({\"✓ OK\" if pages <= 8 else \"✗ OVER LIMIT — cut content\"})')
"
```

---

## D2 — All Figures 300 DPI PDF

```bash
# Verify all figures exist and are PDF
python3 - << 'EOF'
from pathlib import Path

REQUIRED = [
    "paper/figures/fig1_pipeline.pdf",
    "paper/figures/fig2_training_curves.pdf",
    "paper/figures/fig3_curriculum_level.pdf",
    "paper/figures/fig4_results_bar.pdf",
]

print("Figure check:")
all_ok = True
for f in REQUIRED:
    p = Path(f)
    if p.exists():
        size = p.stat().st_size / 1024
        print(f"  ✓ {f} ({size:.1f} KB)")
    else:
        print(f"  ✗ MISSING: {f}")
        all_ok = False

print()
print("D2:", "✅ COMPLETE" if all_ok else "❌ Run Phase B scripts first")
EOF
```

---

## D3 — GitHub Repo Public + README Complete

```bash
cd ~/rl_robotics

# Verify public
git remote -v

# Update README with final results
# (Replace placeholder numbers with actual eval results)

# Verify required files exist
for f in README.md requirements.txt LICENSE .gitignore; do
    [ -f $f ] && echo "✓ $f" || echo "✗ MISSING: $f"
done

# Final commit
git add -A
git commit -m "final: submission-ready code, all results"
git tag v1.0-ral-submission -m "Code as submitted to IEEE RA-L"
git push origin main --tags

echo "Repo URL: https://github.com/Itzlimon22/rl_robotics"
```

---

## D4 — requirements.txt Tested

```bash
# Test requirements.txt on clean environment
conda create -n rl_test python=3.11 -y
conda activate rl_test
pip install -r ~/rl_robotics/requirements.txt

# Verify installation
python3 -c "
import mujoco, gymnasium, stable_baselines3, torch, numpy, matplotlib
print('mujoco:           ', mujoco.__version__)
print('gymnasium:        ', gymnasium.__version__)
print('stable-baselines3:', stable_baselines3.__version__)
print('torch:            ', torch.__version__)
print('numpy:            ', numpy.__version__)
print('matplotlib:       ', matplotlib.__version__)
print('All imports OK ✓')
"

# Run sanity check
cd ~/rl_robotics
python scripts/train.py --mode curriculum --seed 0 --steps 500 --run-name ci_test

# Clean up test env
conda deactivate
conda env remove -n rl_test
```

---

## D5 — Statistical Tests Computed and Reported

```bash
conda activate rl
cd ~/rl_robotics

# Run full statistical analysis
python scripts/results_table.py --set original --stats --ci

# Save to file for reference
python scripts/results_table.py --set original --latex --stats --ci \
    > paper/tables/full_statistics.txt

echo ""
echo "Key numbers to insert in paper:"
echo "  - p-value for CDR vs UDR energy: see above output"
echo "  - Cohen's d effect size: see above output"
echo "  - 95% CI for each condition: see above output"
```

---

## D6 — arXiv Submission (cs.RO)

**Steps:**
1. Go to `arxiv.org` → Submit → New Submission
2. Primary category: **cs.RO** (Robotics)
3. Cross-list: **eess.SY** (Systems and Control)
4. Upload: `paper/main.tex` + all section files + `paper/references.bib` + `paper/figures/` folder
5. Title: exact paper title
6. Authors: your full name
7. Abstract: copy from `paper/sections/abstract.tex` (remove LaTeX commands)
8. Comments: `"8 pages, 4 figures, submitted to IEEE Robotics and Automation Letters"`

**After arXiv accepts (24-48 hours):**

```bash
# Update README with arXiv link
cd ~/rl_robotics
sed -i 's/\[arXiv link — will be added on submission day\]/https:\/\/arxiv.org\/abs\/XXXX.XXXXX/' README.md

git add README.md
git commit -m "docs: add arXiv preprint link"
git push
```

---

## D7 — RA-L Submission via Manuscript Central

**URL:** `mc.manuscriptcentral.com/ral`

**Steps:**
1. Log in → New Submission
2. Select: **IEEE Robotics and Automation Letters**
3. Select: **"Submit to RA-L with option for IROS 2026 presentation"** ← important
4. Upload files:
   - Main PDF: `paper/main.pdf`
   - Source ZIP: `zip -r paper_source.zip paper/main.tex paper/sections/ paper/figures/ paper/references.bib`
   - Cover letter (see template in pipeline document)
5. Suggested Associate Editor area: **Autonomous Systems** or **Marine Robotics**
6. Confirm submission

---

## D8 — Email 8 PhD Supervisors with arXiv Link

```bash
# Target list — personalise each email before sending
TARGET_GROUPS = [
    "Ocean Systems Lab, University of Edinburgh",
    "ACFR, University of Sydney",
    "RSL, ETH Zurich (Hutter group)",
    "Petillot Group, Heriot-Watt University",
    "Clement Group, ENSTA Bretagne",
    "CSAIL, MIT",
    "MBARI (Monterey Bay Aquarium Research Institute)",
    "MarineGym authors — Shuguang Chu et al. (Zhejiang University)",
]

echo "Email template saved in: ~/rl_robotics/paper/email_template.txt"
```

**Create email template:**

```bash
cat > ~/rl_robotics/paper/email_template.txt << 'EMAILEOF'
Subject: PhD Application — AUV Sim-to-Real RL Research (arXiv:XXXX.XXXXX)

Dear Professor [Name],

I am writing to express interest in PhD positions beginning [Sept 2026].

I have independently developed a complete sim-to-real RL pipeline for AUV control,
comparing curriculum and uniform domain randomisation across 9 systematic experiments.
Key finding: PID control collapses from 100% to 3% success under out-of-distribution
fluid parameters, while CDR maintains ~95% success with 6% lower energy consumption.

Preprint: https://arxiv.org/abs/XXXX.XXXXX
Code:     https://github.com/Itzlimon22/rl_robotics

This work is relevant to your group's research on [SPECIFIC TOPIC — personalise this].
The natural next step — hardware validation on a real AUV — is precisely what
collaboration with your group would enable.

I would be grateful for a brief video call at your convenience.

[Your Name] | [Email] | MIST, Bangladesh
EMAILEOF

echo "✓ Email template created"
```

---

## Final Verification Checklist

```bash
python3 - << 'EOF'
from pathlib import Path
import json
import numpy as np

BASE = Path.home() / "rl_research" / "auv"
REPO = Path.home() / "rl_robotics"

checks = {
    # Phase A
    "A1: Original eval JSONs (9)": sum(
        1 for m in ["none","uniform","curriculum"]
        for s in range(3)
        for suffix in ["_v1",""]
        if (BASE / m / f"{m}_seed{s}{suffix}" / "energy_eval_results.json").exists()
    ) >= 9,
    "A4: training_curves.json":    (BASE / "training_curves.json").exists(),
    "A5: curriculum_progression":  (BASE / "curriculum_progression.json").exists(),

    # Phase B
    "B1: fig1_pipeline.pdf":       (REPO / "paper/figures/fig1_pipeline.pdf").exists(),
    "B2: fig2_training_curves.pdf":(REPO / "paper/figures/fig2_training_curves.pdf").exists(),
    "B3: fig3_curriculum.pdf":     (REPO / "paper/figures/fig3_curriculum_level.pdf").exists(),
    "B4: fig4_results_bar.pdf":    (REPO / "paper/figures/fig4_results_bar.pdf").exists(),

    # Phase D
    "D1: paper/main.pdf":          (REPO / "paper/main.pdf").exists(),
    "D3: README.md":               (REPO / "README.md").exists(),
    "D3: requirements.txt":        (REPO / "requirements.txt").exists(),
    "D3: LICENSE":                 (REPO / "LICENSE").exists(),
}

print("=" * 55)
print("FINAL SUBMISSION READINESS CHECK")
print("=" * 55)
all_ok = True
for label, ok in checks.items():
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok: all_ok = False

print()
print("Status:", "🚀 READY TO SUBMIT" if all_ok else "⚠️  Complete remaining items above")
EOF
```

---

*Last updated: after Phase 0-4 complete, master runs seed 0-1 done*
*Work through phases in order — each phase builds on the previous*
*Every command is copy-paste ready. File paths are absolute.*
