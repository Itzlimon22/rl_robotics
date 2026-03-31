# save as scripts/plot_curriculum_level.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"
df = pd.read_csv(BASE / "all_metrics.csv")

df_cdr = df[
    (df["metric"] == "cdr/curriculum_level") & (df["mode"] == "curriculum")
].copy()

TARGET_STEPS = np.linspace(10000, 1000000, 100)
SEED_COLORS = ["#1D9E75", "#0F6E56", "#085041"]

fig, ax = plt.subplots(figsize=(8, 4))

for i, seed in enumerate([0, 1, 2]):
    sub = df_cdr[df_cdr["seed"] == seed]
    if sub.empty:
        continue
    interp = np.interp(TARGET_STEPS, sub["step"].values, sub["value"].values)
    ax.plot(
        TARGET_STEPS, interp, color=SEED_COLORS[i], label=f"Seed {seed}", linewidth=1.5
    )

ax.axhline(
    y=0.833, color="gray", linestyle="--", linewidth=1, label="Maximum level (0.833)"
)
ax.set_xlabel("Training steps", fontsize=12)
ax.set_ylabel("Curriculum level", fontsize=12)
ax.set_title("CDR curriculum expansion over training", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000000)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(BASE / "fig3_curriculum_level.pdf", bbox_inches="tight", dpi=300)
plt.savefig(BASE / "fig3_curriculum_level.png", bbox_inches="tight", dpi=300)
print("Saved fig3_curriculum_level.pdf and .png")
