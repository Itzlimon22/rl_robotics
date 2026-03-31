# save as scripts/convergence_analysis.py
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"
df = pd.read_csv(BASE / "all_metrics.csv")

df_sr = df[df["metric"] == "env/success_rate"].copy()
THRESHOLD = 0.80

print("Steps to reach 80% success rate:")
print("-" * 45)

for mode in ["none", "uniform", "curriculum"]:
    steps_list = []
    for seed in [0, 1, 2]:
        sub = df_sr[(df_sr["mode"] == mode) & (df_sr["seed"] == seed)]
        if sub.empty:
            steps_list.append(None)
            continue
        reached = sub[sub["value"] >= THRESHOLD]
        if reached.empty:
            steps_list.append(None)
            print(f"  {mode} seed{seed}: never reached 80%")
        else:
            step = reached["step"].min()
            steps_list.append(step)
            print(f"  {mode} seed{seed}: step {step:,}")

    valid = [s for s in steps_list if s is not None]
    if valid:
        print(f"  {mode} MEAN: {np.mean(valid):,.0f} steps ± {np.std(valid):,.0f}")
    print()
