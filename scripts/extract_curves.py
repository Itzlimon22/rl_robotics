"""
extract_curves.py — TB to JSON Extractor
================================================================================
Extracts training metrics from master and static runs.
Handles nested SB3 log structures (root vs /eval folders).
"""

import json
import os
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Map TB keys to clean names for your paper
METRIC_MAP = {
    "rollout/ep_rew_mean": "reward",
    "rollout/success_rate": "success",
    "env/goal_dist": "error",
    "cdr/curriculum_level": "curriculum",
    "cdr/success_rate": "rolling_sr",
}


def extract_from_dir(log_dir: Path):
    """Deep search for event files in a directory and extract metrics."""
    data = {}
    # Find all event files in the directory and subdirectories
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return data

    for event_file in event_files:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        for tb_key, clean_name in METRIC_MAP.items():
            if tb_key in ea.Tags()["scalars"]:
                events = ea.Scalars(tb_key)
                if clean_name not in data:
                    data[clean_name] = {"steps": [], "values": []}
                data[clean_name]["steps"].extend([e.step for e in events])
                data[clean_name]["values"].extend([float(e.value) for e in events])

    # Sort by steps in case files were read out of order
    for k in data:
        sort_idx = sorted(
            range(len(data[k]["steps"])), key=lambda i: data[k]["steps"][i]
        )
        data[k]["steps"] = [data[k]["steps"][i] for i in sort_idx]
        data[k]["values"] = [data[k]["values"][i] for i in sort_idx]

    return data


def main():
    base = Path.home() / "rl_research" / "auv"
    results = {
        "master": {"curriculum": [], "uniform": [], "none": []},
        "static": {"curriculum": [], "uniform": [], "none": []},
    }

    print(f"🔍 Searching in: {base}")

    for task in ["master", "static"]:
        for mode in ["curriculum", "uniform", "none"]:
            # Pattern: master_curriculum/master_curriculum_seed0
            mode_folder = base / f"{task}_{mode}"
            if not mode_folder.exists():
                continue

            for seed in [0, 1, 2]:
                run_dir = mode_folder / f"{task}_{mode}_seed{seed}"
                if not run_dir.exists():
                    continue

                print(f"  -> Found: {run_dir.name}")
                # We extract from the whole tree to get rollout + eval data
                run_data = extract_from_dir(run_dir)
                if run_data:
                    results[task][mode].append(run_data)

    out_file = base / "all_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Extraction complete! Saved to {out_file}")


if __name__ == "__main__":
    main()
