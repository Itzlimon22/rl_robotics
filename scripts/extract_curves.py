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

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Please install tensorboard: pip install tensorboard")
    sys.exit(1)

# Default to the local Mac directory structure we've been using
BASE = Path.home() / "rl_research" / "auv"
MODES = ["none", "uniform", "curriculum"]
# We only used seed 0 for the main tests, but you can add 1 and 2 if you ran multiple seeds
SEEDS = [0]
METRICS = ["env/goal_dist", "rollout/ep_rew_mean"]


def extract(path: Path, metric: str):
    ea = EventAccumulator(str(path))
    ea.Reload()
    try:
        events = ea.Scalars(metric)
        return {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    except KeyError:
        print(f"  WARNING: Metric '{metric}' not found in {path.parent.name}")
        return {"steps": [], "values": []}


def main():
    print("Extracting TensorBoard logs...")
    output = {}
    for metric in METRICS:
        output[metric] = {}
        for mode in MODES:
            output[metric][mode] = []
            for seed in SEEDS:
                # E.g., ~/rl_research/auv/obstacle_curriculum/obstacle_curriculum_seed0/tensorboard/
                run_dir_name = f"obstacle_{mode}"
                tb_path = (
                    BASE / run_dir_name / f"{run_dir_name}_seed{seed}" / "tensorboard"
                )

                # Tensorboard puts the actual data in a subfolder, so we grab the first directory inside
                if tb_path.exists() and any(tb_path.iterdir()):
                    actual_tb_log = next(tb_path.iterdir())
                    data = extract(actual_tb_log, metric)
                    output[metric][mode].append(data)
                    print(
                        f"  {mode} seed{seed}: {len(data['steps'])} points for {metric}"
                    )
                else:
                    print(f"  MISSING OR EMPTY: {tb_path}")
                    output[metric][mode].append({"steps": [], "values": []})

    out_path = BASE / "training_curves.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nExtraction complete. Saved to → {out_path}")


if __name__ == "__main__":
    main()
