# save as scripts/plot_training_curves.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"
df = pd.read_csv(BASE / "all_metrics.csv")

# Filter to goal_dist metric
df_dist = df[df["metric"] == "env/goal_dist"].copy()

# Smooth and resample to common steps
TARGET_STEPS = np.linspace(10000, 1000000, 100)
COLORS = {"none": "#E24B4A", "uniform": "#BA7517", "curriculum": "#1D9E75"}
LABELS = {
    "none": "No DR (Baseline)",
    "uniform": "Uniform DR",
    "curriculum": "Curriculum DR (Ours)",
}

fig, ax = plt.subplots(figsize=(8, 4))

for mode in ["none", "uniform", "curriculum"]:
    seed_curves = []
    for seed in [0, 1, 2]:
        sub = df_dist[(df_dist["mode"] == mode) & (df_dist["seed"] == seed)]
        if sub.empty:
            continue
        # Interpolate to common steps
        interp = np.interp(TARGET_STEPS, sub["step"].values, sub["value"].values)
        seed_curves.append(interp)

    if not seed_curves:
        continue

    arr = np.array(seed_curves)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    ax.plot(TARGET_STEPS, mean, color=COLORS[mode], label=LABELS[mode], linewidth=1.5)
    ax.fill_between(
        TARGET_STEPS, mean - std, mean + std, alpha=0.15, color=COLORS[mode]
    )

ax.set_xlabel("Training steps", fontsize=12)
ax.set_ylabel("Mean goal distance (m)", fontsize=12)
ax.set_title("Training performance across conditions", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000000)
plt.tight_layout()
plt.savefig(BASE / "fig2_training_curves.pdf", bbox_inches="tight", dpi=300)
plt.savefig(BASE / "fig2_training_curves.png", bbox_inches="tight", dpi=300)
print("Saved fig2_training_curves.pdf and .png")
