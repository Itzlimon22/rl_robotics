import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# --- Publication Settings ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
        "figure.autolayout": True,
    }
)

COLORS = {"none": "#D62728", "uniform": "#FF7F0E", "curriculum": "#2CA02C"}
LABELS = {"none": "Naive SAC", "uniform": "Uniform DR", "curriculum": "CDR (Ours)"}


def plot_final_analysis(task="master"):
    base = Path.home() / "rl_research" / "auv"
    with open(base / "all_results.json") as f:
        data = json.load(f)[task]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("reward", "Mean Reward"),
        ("success", "Success Rate"),
        ("curriculum", "CDR Progress"),
    ]

    for i, (m_key, title) in enumerate(metrics):
        ax = axes[i]
        for mode in ["none", "uniform", "curriculum"]:
            seeds = data[mode]
            if not seeds or m_key not in seeds[0]:
                continue

            # Aggregate across seeds
            min_len = min(len(s[m_key]["values"]) for s in seeds)
            vals = np.array([s[m_key]["values"][:min_len] for s in seeds])
            steps = np.array(seeds[0][m_key]["steps"][:min_len]) / 1e6  # Millions

            mean = gaussian_filter1d(np.mean(vals, axis=0), sigma=2)
            std = np.std(vals, axis=0) if len(seeds) > 1 else np.zeros_like(mean)

            ax.plot(steps, mean, label=LABELS[mode], color=COLORS[mode], lw=1.5)
            if len(seeds) > 1:
                ax.fill_between(
                    steps,
                    mean - std,
                    mean + std,
                    color=COLORS[mode],
                    alpha=0.15,
                    edgecolor="none",
                )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Timesteps ($10^6$)")
        if i == 0:
            ax.legend(frameon=False)
        if i == 1:
            ax.set_ylim(0, 1.05)  # Success rate is 0 to 1

    plt.savefig(base / f"figure2_{task}_results.pdf", bbox_inches="tight")
    print(f"✅ Created Figure 2 (PDF) for {task} task.")


if __name__ == "__main__":
    plot_final_analysis("master")
