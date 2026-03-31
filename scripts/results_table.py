"""
results_table.py — Aggregate eval results into paper Table 2
Usage: python scripts/results_table.py
"""

import json
import numpy as np
from pathlib import Path

MODES = ["none", "uniform", "curriculum"]
SEEDS = [0, 1, 2]
LABELS = {"none": "Naive SAC", "uniform": "Uniform DR", "curriculum": "CDR (Ours)"}


def load_results(base: Path):
    table = {}
    for mode in MODES:
        results = []
        for seed in SEEDS:
            p = base / mode / f"{mode}_seed{seed}" / "test_eval_results.json"
            if p.exists():
                with open(p) as f:
                    results.append(json.load(f))
            else:
                print(f"  MISSING: {p}")
        table[mode] = results
    return table


def main():
    on_colab = Path("/content/drive/MyDrive").exists()
    base = (
        Path("/content/drive/MyDrive/rl_research/auv")
        if on_colab
        else Path.home() / "rl_research" / "auv"
    )

    table = load_results(base)

    print(
        "\n=== PAPER TABLE 2 — Transfer Evaluation (held-out test distribution) ===\n"
    )
    print(
        f"{'Method':<20} {'Success Rate':>15} {'Mean Reward':>14} {'Mean Dist (m)':>14}"
    )
    print("-" * 65)

    for mode in MODES:
        results = table[mode]
        if not results:
            print(f"{LABELS[mode]:<20}  NO DATA")
            continue
        sr = np.mean([r["success_rate"] for r in results])
        sr_s = np.std([r["success_rate"] for r in results])
        rew = np.mean([r["mean_reward"] for r in results])
        rew_s = np.std([r["mean_reward"] for r in results])
        dist = np.mean([r["mean_dist"] for r in results])
        dist_s = np.std([r["mean_dist"] for r in results])
        print(
            f"{LABELS[mode]:<20} "
            f"{sr * 100:>8.1f}±{sr_s * 100:.1f}%   "
            f"{rew:>8.1f}±{rew_s:.1f}   "
            f"{dist:>8.2f}±{dist_s:.2f}m"
        )

    print()

    # LaTeX version
    print("=== LaTeX ===\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Success Rate (\%) & Mean Reward & Mean Dist (m) \\")
    print(r"\hline")
    for mode in MODES:
        results = table[mode]
        if not results:
            continue
        sr = np.mean([r["success_rate"] for r in results]) * 100
        sr_s = np.std([r["success_rate"] for r in results]) * 100
        rew = np.mean([r["mean_reward"] for r in results])
        rew_s = np.std([r["mean_reward"] for r in results])
        dist = np.mean([r["mean_dist"] for r in results])
        dist_s = np.std([r["mean_dist"] for r in results])
        bold = mode == "curriculum"
        name = f"\\textbf{{{LABELS[mode]}}}" if bold else LABELS[mode]
        print(
            f"{name} & "
            f"${sr:.1f} \\pm {sr_s:.1f}$ & "
            f"${rew:.1f} \\pm {rew_s:.1f}$ & "
            f"${dist:.2f} \\pm {dist_s:.2f}$ \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Transfer evaluation on held-out test distribution.}")
    print(r"\label{tab:transfer_results}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
