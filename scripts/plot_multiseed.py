import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_data_from_npz(npz_path):
    """Extracts evaluation data from evaluations.npz file."""
    try:
        data = np.load(npz_path)
        # timesteps: [10000, 20000, ...]
        # results: [num_evals, num_episodes] -> we take the mean across episodes
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rewards = np.mean(results, axis=1)
        return timesteps, mean_rewards
    except Exception:
        return None, None


def get_data_from_tb(log_dir):
    """Extracts data from TensorBoard events file."""
    tag = "rollout/ep_rew_mean"
    event_files = glob.glob(str(log_dir / "**/events.out.tfevents.*"), recursive=True)
    if not event_files:
        return None, None
    event_file = max(event_files, key=os.path.getctime)
    try:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return None, None
        events = ea.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])
    except Exception:
        return None, None


def main():
    base_path = Path.home() / "rl_research" / "auv"
    modes = ["none", "uniform", "curriculum"]
    colors = {"none": "#e74c3c", "uniform": "#f39c12", "curriculum": "#3498db"}
    data_dict = {m: [] for m in modes}

    print(f"📂 Scanning AUV Research Vault: {base_path}")

    # 1. Recursive Scan for all subdirectories
    all_run_dirs = [
        d for d in base_path.rglob("*") if d.is_dir() and "seed" in d.name.lower()
    ]

    for run_dir in all_run_dirs:
        # Identify mode
        active_mode = next((m for m in modes if m in str(run_dir).lower()), None)
        if not active_mode:
            continue

        # Plan A: Try TensorBoard
        steps, values = get_data_from_tb(run_dir)

        # Plan B: Try NPZ if TB failed
        if steps is None:
            npz_file = run_dir / "eval" / "evaluations.npz"
            if npz_file.exists():
                steps, values = get_data_from_npz(npz_file)

        if steps is not None and len(values) > 2:
            common_steps = np.linspace(0, 1_000_000, 1000)
            interp_val = np.interp(common_steps, steps, values)
            smoothed_val = gaussian_filter1d(interp_val, sigma=5)
            data_dict[active_mode].append(smoothed_val)
            print(f" ✅ Loaded {active_mode.upper()} from {run_dir.name}")

    # 2. Plotting logic
    plt.figure(figsize=(10, 6))
    plotted = False
    for mode, value_list in data_dict.items():
        if not value_list:
            continue
        plotted = True
        stacked = np.vstack(value_list)
        mean_vals, common_axis = (
            np.mean(stacked, axis=0),
            np.linspace(0, 1_000_000, 1000),
        )

        if len(value_list) > 1:
            std_vals = np.std(stacked, axis=0)
            plt.fill_between(
                common_axis,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=colors[mode],
                alpha=0.15,
            )
        plt.plot(
            common_axis,
            mean_vals,
            label=f"SAC-{mode.capitalize()} (N={len(value_list)})",
            color=colors[mode],
            lw=2,
        )

    if not plotted:
        print("❌ No data found in either TB or NPZ formats.")
        return

    plt.title(
        "AUV Learning Convergence (Multi-Format Support)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = (
        Path.home()
        / "rl_research"
        / "paper_assets"
        / "figures"
        / "master_learning_curves.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\n🚀 SUCCESS! Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
