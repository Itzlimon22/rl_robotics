import json
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_run(run_dir: Path):
    data = {}

    # 1. Get Goal Dist from TensorBoard (Fixed to avoid loop-overwriting)
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        tb_dir = run_dir  # Fallback if logs are directly in the run folder

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()

    if "scalars" in ea.Tags() and "env/goal_dist" in ea.Tags()["scalars"]:
        events = ea.Scalars("env/goal_dist")
        data["env/goal_dist"] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }

    # 2. Get True Rewards from evaluations.npz (Brilliant move!)
    npz_file = run_dir / "eval" / "evaluations.npz"
    if npz_file.exists():
        npz = np.load(npz_file)
        data["eval/mean_reward"] = {
            "steps": npz["timesteps"].tolist(),
            "values": npz["results"].mean(axis=1).tolist(),
        }
    else:
        print(f"    [Warning] Missing NPZ: {npz_file}")

    return data


def main():
    # Auto-detect Colab vs Mac
    if Path("/content/drive/MyDrive").exists():
        base = Path("/content/drive/MyDrive/rl_research/auv")
    else:
        base = Path.home() / "rl_research" / "auv"

    # Route output strictly to the paper/data directory per your rule
    out_dir = base / "paper" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "training_curves.json"

    output = {"env/goal_dist": {}, "eval/mean_reward": {}}
    modes = ["none", "uniform", "curriculum"]

    print(f"🔍 Searching in: {base}")

    for mode in modes:
        output["env/goal_dist"][mode] = []
        output["eval/mean_reward"][mode] = []

        for seed in [0, 1, 2]:
            # FIXED: Added the "tracking_" prefix to match your actual run names
            run_name = f"tracking_{mode}_seed{seed}"

            # Check both possible SB3 save structures
            run_dir = base / "tracking" / run_name
            if not run_dir.exists():
                run_dir = base / mode / run_name  # Fallback

            if run_dir.exists():
                print(f"  -> Extracting: {run_dir.name}")
                run_data = extract_run(run_dir)

                if "env/goal_dist" in run_data:
                    output["env/goal_dist"][mode].append(run_data["env/goal_dist"])
                if "eval/mean_reward" in run_data:
                    output["eval/mean_reward"][mode].append(
                        run_data["eval/mean_reward"]
                    )
            else:
                print(f"    [Skip] Not found: {run_dir}")

    with open(out_file, "w") as f:
        json.dump(output, f)
    print(f"\n✅ Extracted and saved to {out_file}")


if __name__ == "__main__":
    main()
