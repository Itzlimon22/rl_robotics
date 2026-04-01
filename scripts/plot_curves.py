"""
plot_curves.py — Plot training curves for CoRL paper
====================================================
Generates Figure 2 (training curves) from extracted TensorBoard data.

Usage:
    python scripts/plot_curves.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# Publication-ready formatting
mpl.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    }
)

COLORS = {
    "none": "#E24B4A",  # Red
    "uniform": "#BA7517",  # Orange/Gold
    "curriculum": "#1D9E75",  # Green
}
LABELS = {
    "none": "Naive SAC",
    "uniform": "Uniform DR",
    "curriculum": "CDR (ours)",
}
SMOOTH = 30  # Moving average window size for cleaner lines


def plot_metric(curves, metric_key, ylabel, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode in ["none", "uniform", "curriculum"]:
        seeds = curves[metric_key][mode]
        valid = [s for s in seeds if len(s["values"]) > 0]
        if not valid:
            continue

        # Align lengths in case some runs stopped slightly early
        min_len = min(len(s["values"]) for s in valid)
        vals = np.array([s["values"][:min_len] for s in valid])
        steps = np.array(valid[0]["steps"][:min_len])

        mean = uniform_filter1d(np.mean(vals, axis=0), size=SMOOTH)

        # Only plot standard deviation shading if you ran multiple seeds
        if len(valid) > 1:
            std = uniform_filter1d(np.std(vals, axis=0), size=SMOOTH)
            ax.fill_between(
                steps / 1e6, mean - std, mean + std, alpha=0.15, color=COLORS[mode]
            )

        ax.plot(
            steps / 1e6, mean, color=COLORS[mode], label=LABELS[mode], linewidth=2.0
        )

    ax.set_xlabel("Training steps (×10⁶)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0, 1.0)  # Assuming 1M steps
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Save to the base AUV folder
    base = Path.home() / "rl_research" / "auv"
    out_path = base / out_name
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close()


def main():
    curves_path = Path.home() / "rl_research" / "auv" / "training_curves.json"
    if not curves_path.exists():
        print("Error: training_curves.json not found. Run extract_curves.py first!")
        return

    with open(curves_path) as f:
        curves = json.load(f)

    print("Generating plots...")
    plot_metric(curves, "env/goal_dist", "Mean Goal Distance (m)", "fig2_goal_dist.pdf")
    plot_metric(curves, "rollout/ep_rew_mean", "Mean Episode Reward", "fig2_reward.pdf")
    print("Done! Your paper figures are ready.")


if __name__ == "__main__":
    main()
