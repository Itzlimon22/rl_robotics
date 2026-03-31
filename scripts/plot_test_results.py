# save as scripts/plot_test_results.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "rl_research" / "auv"

# Your eval results — fill in from your eval output
results = {
    "none": [28.0, 62.0, 96.0],
    "uniform": [100.0, 100.0, 100.0],
    "curriculum": [94.0, 100.0, 90.0],
}

labels = ["No DR\n(Baseline)", "Uniform DR\n(Baseline)", "Curriculum DR\n(Ours)"]
colors = ["#E24B4A", "#BA7517", "#1D9E75"]
x = np.arange(len(labels))

means = [np.mean(results[k]) for k in ["none", "uniform", "curriculum"]]
stds = [np.std(results[k]) for k in ["none", "uniform", "curriculum"]]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    x,
    means,
    yerr=stds,
    capsize=6,
    color=colors,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.5,
    width=0.5,
)

# Value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        mean + std + 1.5,
        f"{mean:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Success rate on held-out test distribution (%)", fontsize=11)
ax.set_ylim(0, 120)
ax.axhline(
    y=100, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="100% ceiling"
)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("Zero-shot transfer to held-out fluid dynamics regimes", fontsize=12)
plt.tight_layout()
plt.savefig(BASE / "fig4_test_results.pdf", bbox_inches="tight", dpi=300)
plt.savefig(BASE / "fig4_test_results.png", bbox_inches="tight", dpi=300)
print("Saved fig4_test_results.pdf and .png")
