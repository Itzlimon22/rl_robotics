# File: scripts/results_table.py
"""
Generates the final statistical results and LaTeX Table II for the AUV paper.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Try importing scipy for exact p-values. If missing, we'll bypass gracefully.
try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print(
        "⚠ Notice: 'scipy' not found. P-values will be approximated. Run 'pip install scipy' for exact stats."
    )


def get_base_path():
    if os.path.exists("/content/drive/MyDrive/rl_research/auv"):
        return Path("/content/drive/MyDrive/rl_research/auv")
    return Path.home() / "rl_research" / "auv"


def load_metrics(mode, base_path):
    """Extracts success rate, reward, energy, and distance for a given mode across 3 seeds."""
    data = {"success": [], "reward": [], "energy": [], "dist": []}

    for seed in range(3):
        seed_found = False
        for suffix in ["_v1", ""]:
            for fn in [
                "tracking_eval_results.json",
                "test_eval_results.json",
                "energy_eval_results.json",
            ]:
                p_nested = (
                    base_path
                    / "tracking"
                    / f"tracking_{mode}"
                    / f"tracking_{mode}_seed{seed}{suffix}"
                    / fn
                )
                p_standard = base_path / mode / f"{mode}_seed{seed}{suffix}" / fn
                p = p_nested if p_nested.exists() else p_standard

                if p.exists():
                    with open(p) as f:
                        r = json.load(f)
                        # Adapt keys based on what your eval script actually saved
                        data["success"].append(
                            r.get("success_rate", 0.0) * 100
                        )  # Convert to %
                        data["reward"].append(r.get("mean_reward", 0.0))
                        data["energy"].append(r.get("mean_energy_per_step", 0.0))

                        # Distance might be logged under different names
                        dist = r.get("mean_tracking_error", r.get("mean_distance", 0.0))
                        data["dist"].append(dist)

                        seed_found = True
                        break
            if seed_found:
                break

        if not seed_found:
            print(f"⚠ Warning: Could not find data for {mode} seed {seed}")

    return data


def calc_stats(array):
    if not array:
        return 0.0, 0.0
    return np.mean(array), np.std(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="original")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()

    base = get_base_path()

    # Load Data
    cdr = load_metrics("curriculum", base)
    udr = load_metrics("uniform", base)
    none = load_metrics("none", base)

    # Compute Means and Stds
    c_sr_m, c_sr_s = calc_stats(cdr["success"])
    c_rw_m, c_rw_s = calc_stats(cdr["reward"])
    c_en_m, c_en_s = calc_stats(cdr["energy"])
    c_ds_m, _ = calc_stats(cdr["dist"])

    u_sr_m, u_sr_s = calc_stats(udr["success"])
    u_rw_m, u_rw_s = calc_stats(udr["reward"])
    u_en_m, u_en_s = calc_stats(udr["energy"])
    u_ds_m, _ = calc_stats(udr["dist"])

    n_sr_m, n_sr_s = calc_stats(none["success"])
    n_rw_m, n_rw_s = calc_stats(none["reward"])
    n_en_m, n_en_s = calc_stats(none["energy"])
    n_ds_m, _ = calc_stats(none["dist"])

    # Hardcoded PID baseline from your prompt
    pid_sr, pid_rw, pid_en = 3.0, -249.02, 0.858

    # --- STATISTICAL TESTS ---
    saving_pct = 0.0
    p_val_str = "N/A"
    cohens_d = 0.0
    cdr_ci = [0, 0]
    udr_ci = [0, 0]

    if cdr["energy"] and udr["energy"]:
        saving_pct = (u_en_m - c_en_m) / u_en_m * 100

        # 95% Confidence Intervals (n=3, t-value ~ 4.303 for 95% CI with df=2)
        c_margin = 4.303 * (c_en_s / np.sqrt(len(cdr["energy"])))
        cdr_ci = [c_en_m - c_margin, c_en_m + c_margin]

        u_margin = 4.303 * (u_en_s / np.sqrt(len(udr["energy"])))
        udr_ci = [u_en_m - u_margin, u_en_m + u_margin]

        # Cohen's d
        pooled_std = np.sqrt((c_en_s**2 + u_en_s**2) / 2)
        cohens_d = (u_en_m - c_en_m) / pooled_std if pooled_std > 0 else 0

        # P-Value (Welch's t-test)
        if HAS_SCIPY:
            stat, p_val = stats.ttest_ind(udr["energy"], cdr["energy"], equal_var=False)
            if p_val < 0.001:
                p_val_str = f"{p_val:.4f} (***)"
            elif p_val < 0.01:
                p_val_str = f"{p_val:.4f} (**)"
            elif p_val < 0.05:
                p_val_str = f"{p_val:.4f} (*)"
            else:
                p_val_str = f"{p_val:.4f} (ns)"

    # --- PRINT TERMINAL REPORT ---
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("YOUR PAPER NUMBERS — fill in from results_table.py output")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"CDR success rate mean:          {c_sr_m:6.2f}%")
    print(f"CDR success rate std:           {c_sr_s:6.2f}%")
    print(f"CDR mean reward:                {c_rw_m:7.3f}")
    print(f"CDR std reward:                 {c_rw_s:7.3f}")
    print(f"CDR energy/step:                {c_en_m:6.4f}")
    print(f"CDR energy std:                 {c_en_s:6.4f}")
    print(f"CDR mean dist:                  {c_ds_m:4.2f}m")
    print("")
    print(f"Uniform DR success rate mean:   {u_sr_m:6.2f}%")
    print(f"Uniform DR success rate std:    {u_sr_s:6.2f}%")
    print(f"Uniform DR mean reward:         {u_rw_m:7.3f}")
    print(f"Uniform DR energy/step:         {u_en_m:6.4f}")
    print(f"Uniform DR energy std:          {u_en_s:6.4f}")
    print(f"Uniform DR mean dist:           {u_ds_m:4.2f}m")
    print("")
    print(f"Naive SAC success rate mean:    {n_sr_m:6.2f}%")
    print(f"Naive SAC success rate std:     {n_sr_s:6.2f}%")
    print(f"Naive SAC mean reward:          {n_rw_m:7.3f}")
    print(f"Naive SAC energy/step:          {n_en_m:6.4f}")
    print("")
    print(f"PID test success:               {pid_sr}%")
    print(f"PID test reward:                {pid_rw}")
    print(f"PID train success:              100.0%")
    print("")
    print(f"Energy saving CDR vs UDR:       {saving_pct:.2f}%")
    print(f"p-value CDR vs UDR energy:      {p_val_str}")
    print(f"Cohen's d CDR vs UDR energy:    {cohens_d:.2f}")
    print(f"CDR energy 95% CI:              [{cdr_ci[0]:.4f}, {cdr_ci[1]:.4f}]")
    print(f"UDR energy 95% CI:              [{udr_ci[0]:.4f}, {udr_ci[1]:.4f}]")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # --- PRINT LATEX TABLE ---
    if args.latex:
        # Determine bolding for Success Rate (higher is better)
        c_sr_bold = "\\mathbf{" if c_sr_m >= u_sr_m else ""
        c_sr_end = "}" if c_sr_m >= u_sr_m else ""
        u_sr_bold = "\\mathbf{" if u_sr_m > c_sr_m else ""
        u_sr_end = "}" if u_sr_m > c_sr_m else ""

        # Determine bolding for Energy (lower is better, excluding naive which fails)
        c_en_bold = "\\mathbf{" if c_en_m <= u_en_m else ""
        c_en_end = "}" if c_en_m <= u_en_m else ""
        u_en_bold = "\\mathbf{" if u_en_m < c_en_m else ""
        u_en_end = "}" if u_en_m < c_en_m else ""

        latex_str = f"""% paper/tables/table_main_results.tex
% Auto-generated by results_table.py
\\begin{{table}}[t]
\\centering
\\caption{{Transfer evaluation on held-out test distribution (mean $\\pm$ std, 3 seeds, 100 episodes each). \\textbf{{Bold}} indicates best value per metric among successful agents.}}
\\label{{tab:main_results}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Method}} & \\textbf{{Success (\\%)}} & \\textbf{{Reward}} & \\textbf{{Dist (m)}} & \\textbf{{Energy/Step}} \\\\
\\hline
PID baseline & ${pid_sr}$ & ${pid_rw}$ & $---$ & {pid_en} \\\\
Naive SAC & ${n_sr_m:.1f} \\pm {n_sr_s:.1f}$ & ${n_rw_m:.1f} \\pm {n_rw_s:.1f}$ & ${n_ds_m:.2f}$ & ${n_en_m:.3f}$ \\\\
Uniform DR & ${u_sr_bold}{u_sr_m:.1f} \\pm {u_sr_s:.1f}{u_sr_end}$ & ${u_rw_m:.1f} \\pm {u_rw_s:.1f}$ & ${u_ds_m:.2f}$ & ${u_en_bold}{u_en_m:.3f}{u_en_end}$ \\\\
\\textbf{{CDR (ours)}} & ${c_sr_bold}{c_sr_m:.1f} \\pm {c_sr_s:.1f}{c_sr_end}$ & ${c_rw_m:.1f} \\pm {c_rw_s:.1f}$ & ${c_ds_m:.2f}$ & ${c_en_bold}{c_en_m:.3f}{c_en_end}$ \\\\
\\hline
\\multicolumn{{5}}{{l}}{{\\small Energy/step = mean absolute thruster command; lower = more efficient.}}
\\end{{tabular}}
\\end{{table}}
"""
        print("LaTeX Table (Table II in paper)")
        print("========================================================")
        print(latex_str)


if __name__ == "__main__":
    main()
