"""
extract_curves.py — TB & NPZ to JSON Extractor
================================================================================
Extracts goal_dist from TensorBoard and true evaluation rewards from evaluations.npz.
"""

import json
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Map TB keys to clean names (only for TB metrics now)
METRIC_MAP = {
    "env/goal_dist": "error",
    "cdr/curriculum_level": "curriculum",
}


def extract_from_dir(run_dir: Path):
    """Extracts from both TensorBoard and evaluations.npz"""
    data = {}

    # 1. Get Goal Dist / Training Metrics from TensorBoard
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    for event_file in event_files:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])

        for tb_key, clean_name in METRIC_MAP.items():
            if tb_key in scalars:
                events = ea.Scalars(tb_key)
                if clean_name not in data:
                    data[clean_name] = {"steps": [], "values": []}
                data[clean_name]["steps"].extend([e.step for e in events])
                data[clean_name]["values"].extend([float(e.value) for e in events])

    # Sort TB data to prevent matplotlib zigzag rendering
    for k in data:
        sort_idx = np.argsort(data[k]["steps"])
        data[k]["steps"] = [data[k]["steps"][i] for i in sort_idx]
        data[k]["values"] = [data[k]["values"][i] for i in sort_idx]

    # 2. Get True Evaluation Rewards from evaluations.npz
    eval_npz = run_dir / "eval" / "evaluations.npz"
    if eval_npz.exists():
        npz_data = np.load(eval_npz)
        # timesteps shape: (N,)
        steps = npz_data["timesteps"].tolist()
        # results shape: (N, n_eval_episodes) -> mean across episodes
        mean_rewards = npz_data["results"].mean(axis=1).tolist()

        data["reward"] = {"steps": steps, "values": mean_rewards}
    else:
        print(f"    [Warning] No evaluations.npz found in {run_dir.name}")

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
            mode_folder = base / f"{task}_{mode}"
            if not mode_folder.exists():
                continue

            for seed in [0, 1, 2]:
                run_dir = mode_folder / f"{task}_{mode}_seed{seed}"
                if not run_dir.exists():
                    continue

                print(f"  -> Found: {run_dir.name}")
                run_data = extract_from_dir(run_dir)
                if run_data:
                    results[task][mode].append(run_data)

    out_file = base / "all_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Extraction complete! Saved to {out_file}")


if __name__ == "__main__":
    main()
