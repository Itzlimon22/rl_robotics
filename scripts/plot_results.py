"""
plot_results.py — Publication-Ready RL Data Extraction
================================================================================
Extracts raw data from TensorBoard logs and plots a high-resolution, IEEE-style
dual-axis graph showing Success Rate vs. Curriculum Level.
"""

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth_data(scalars, weight=0.85):
    """
    Applies an Exponential Moving Average (EMA) to smooth the data,
    mimicking the standard TensorBoard smoothing algorithm.
    """
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    # 1. Locate the TensorBoard event file recursively
    base_log_dir = os.path.expanduser("~/rl_research/auv/master_curriculum")

    # Search for all tfevents files anywhere inside the base directory
    search_pattern = os.path.join(base_log_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(search_pattern, recursive=True)

    if not event_files:
        print(f"Error: No TensorBoard logs found anywhere inside {base_log_dir}")
        print("Are you sure the training script has started writing logs?")
        return

    # Automatically grab the most recently modified log file
    event_file = max(event_files, key=os.path.getctime)
    print(f"🔍 Found log file! Loading data from:\n{event_file}")

    # 2. Extract Data using EventAccumulator
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Define the exact tags we want to pull from your logs
    sr_tag = "rollout/success_rate"
    curr_tag = "cdr/curriculum_level"

    # Check if tags exist in the log
    available_tags = ea.Tags().get("scalars", [])
    if sr_tag not in available_tags or curr_tag not in available_tags:
        print(f"Error: Found the log, but couldn't find '{sr_tag}' or '{curr_tag}'.")
        print("Available tags:", available_tags)
        return

    # Extract steps and values
    sr_events = ea.Scalars(sr_tag)
    curr_events = ea.Scalars(curr_tag)

    steps = [e.step for e in sr_events]
    sr_raw = [e.value for e in sr_events]
    curr_raw = [e.value for e in curr_events]

    # Smooth the success rate for better readability in the paper
    sr_smoothed = smooth_data(sr_raw, weight=0.90)

    # 3. Plotting (Academic Style)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot 1: Success Rate (Left Y-Axis)
    color1 = "#1f77b4"  # Standard Blue
    ax1.set_xlabel("Training Steps", fontweight="bold")
    ax1.set_ylabel("Rolling Success Rate", color=color1, fontweight="bold")

    # Plot raw data faintly in the background
    ax1.plot(steps, sr_raw, color=color1, alpha=0.2, label="Raw Success Rate")
    # Plot smoothed data clearly
    ax1.plot(
        steps, sr_smoothed, color=color1, linewidth=2, label="Smoothed Success Rate"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Curriculum Level (Right Y-Axis)
    ax2 = ax1.twinx()
    color2 = "#ff7f0e"  # Standard Orange
    ax2.set_ylabel("Curriculum Difficulty Level", color=color2, fontweight="bold")

    # Use a step plot for curriculum since it changes in discrete jumps
    ax2.step(
        steps,
        curr_raw,
        color=color2,
        linewidth=2.5,
        linestyle="--",
        label="Curriculum Level",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim([0, 1.05])

    # 4. Professional Polish: Formatting
    plt.title(
        "Adaptive Curriculum Progression vs. Agent Success Rate",
        fontweight="bold",
        pad=15,
    )

    # Combine legends from both axes
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

    # 5. Save High-Res Output
    output_path = os.path.expanduser(
        "~/rl_research/auv/master_curriculum/curriculum_learning_curve.png"
    )
    plt.savefig(output_path, format="png", bbox_inches="tight")
    print(f"✅ Success! Publication-ready graph saved to:\n{output_path}")


if __name__ == "__main__":
    main()
