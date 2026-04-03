"""
plot_multiseed.py — Publication-Ready Multi-Seed RL Data Aggregation
================================================================================
Extracts data from multiple TensorBoard logs, calculates the mean and standard
deviation across seeds, and plots an academic dual-axis graph.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import interp1d


def extract_tb_data(log_dir, tag):
    """Finds the log file in a directory and extracts steps and values for a tag."""
    search_pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(search_pattern, recursive=True)
    if not event_files:
        return None, None

    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def main():
    base_dir = os.path.expanduser("~/rl_research/auv/master_curriculum")
    seed_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if "seed" in d and os.path.isdir(os.path.join(base_dir, d))
    ]

    print(f"🔍 Found {len(seed_dirs)} seed directories.")

    sr_tag = "rollout/success_rate"
    curr_tag = "cdr/curriculum_level"

    all_sr_interp = []
    all_curr_interp = []

    # Create a common X-axis (steps) to interpolate all seeds onto
    common_steps = np.linspace(0, 1000000, num=1000)

    for s_dir in seed_dirs:
        steps_sr, vals_sr = extract_tb_data(s_dir, sr_tag)
        steps_cr, vals_cr = extract_tb_data(s_dir, curr_tag)

        if steps_sr is not None and steps_cr is not None:
            # Interpolate so all seeds share the exact same step intervals
            f_sr = interp1d(
                steps_sr,
                vals_sr,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            f_cr = interp1d(
                steps_cr,
                vals_cr,
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )

            all_sr_interp.append(f_sr(common_steps))
            all_curr_interp.append(f_cr(common_steps))

    if not all_sr_interp:
        print("Error: No valid data found in seed directories.")
        return

    # Calculate Mean and Standard Deviation
    sr_mean = np.mean(all_sr_interp, axis=0)
    sr_std = np.std(all_sr_interp, axis=0)

    curr_mean = np.mean(all_curr_interp, axis=0)

    # Plotting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Success Rate (Blue)
    color1 = "#1f77b4"
    ax1.set_xlabel("Training Steps", fontweight="bold")
    ax1.set_ylabel("Mean Success Rate", color=color1, fontweight="bold")
    ax1.plot(
        common_steps, sr_mean, color=color1, linewidth=2.5, label="Success Rate (Mean)"
    )
    # Add shaded variance
    ax1.fill_between(
        common_steps,
        sr_mean - sr_std,
        sr_mean + sr_std,
        color=color1,
        alpha=0.2,
        label="± 1 Std Dev",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 1.05])

    # Curriculum Level (Orange)
    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.set_ylabel("Mean Curriculum Level", color=color2, fontweight="bold")
    ax2.plot(
        common_steps,
        curr_mean,
        color=color2,
        linewidth=2.5,
        linestyle="--",
        label="Curriculum Level",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim([0, 1.05])

    plt.title(
        f"Robustness across {len(all_sr_interp)} Seeds: CDR Success vs. Difficulty",
        fontweight="bold",
        pad=15,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower right",
        frameon=True,
        shadow=True,
    )

    fig.tight_layout()
    output_path = os.path.join(base_dir, "multiseed_learning_curve.png")
    plt.savefig(output_path, format="png", bbox_inches="tight")
    print(f"✅ Success! Multi-seed graph saved to: {output_path}")


if __name__ == "__main__":
    main()
