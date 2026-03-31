# save as scripts/extract_tb.py
import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path.home() / "rl_research" / "auv"
METRICS = ["env/goal_dist", "env/success_rate", "cdr/curriculum_level"]
MODES = ["none", "uniform", "curriculum"]
SEEDS = [0, 1, 2]

records = []
for mode in MODES:
    for seed in SEEDS:
        tb_path = BASE / mode / f"{mode}_seed{seed}" / "tensorboard"
        if not tb_path.exists():
            print(f"Missing: {tb_path}")
            continue
        # Find the event file inside the run subdirectory
        subdirs = list(tb_path.iterdir())
        if not subdirs:
            continue
        ea = EventAccumulator(str(subdirs[0]))
        ea.Reload()
        available = ea.Tags()["scalars"]
        for metric in METRICS:
            if metric not in available:
                continue
            events = ea.Scalars(metric)
            for e in events:
                records.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "metric": metric,
                        "step": e.step,
                        "value": e.value,
                    }
                )

df = pd.DataFrame(records)
df.to_csv(BASE / "all_metrics.csv", index=False)
print(f"Saved {len(df)} records to all_metrics.csv")
print(df.groupby(["mode", "metric"])["step"].max())
