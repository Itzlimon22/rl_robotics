"""
plot_ablation.py — Figure 5: ablation horizontal bar chart
===========================================================
Shows which single physics parameter provides the most transfer benefit.
Compares all 5 ablation conditions + full Uniform DR + No DR baselines.

Output: paper/figures/fig5_ablation.pdf
        Drive: /content/drive/MyDrive/rl_research/auv/paper/figures/fig5_ablation.pdf

Usage: python scripts/plot_ablation.py
"""

import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    }
)

BASE = Path.home() / "rl_research" / "auv"
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ablation results ─────────────────────────────────────────────────────
summary_path = BASE / "ablation_summary.json"
if not summary_path.exists():
    raise FileNotFoundError(
        f"{summary_path} not found.\nRun: python scripts/eval_ablation.py"
    )

with open(summary_path) as f:
    ablation = json.load(f)


# ── Load baseline results for reference lines ──────────────────────────────
def load_baseline(mode, n_seeds=3):
    """Load energy_eval_results.json for a condition across seeds."""
    srs = []
    for seed in range(n_seeds):
        for suffix in ["_v1", ""]:
            p = BASE / mode / f"{mode}_seed{seed}{suffix}" / "energy_eval_results.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                srs.append(r["success_rate"] * 100)
                break
    return np.mean(srs) if srs else None


udr_sr = load_baseline("uniform")
naive_sr = load_baseline("none")
pid_sr = 3.0  # fixed from eval

# ── Sort ablation by mean success rate (ascending — worst at bottom) ──────────
ablation.sort(key=lambda x: x["sr_mean"])

labels = [r["label"] for r in ablation]
means = [r["sr_mean"] for r in ablation]
stds = [r["sr_std"] for r in ablation]
n_seeds = [r.get("n_seeds", 1) for r in ablation]

# Color: ablation bars in professional blue, baselines as reference lines
COLORS = ["#2E86AB"] * len(ablation)

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
y = np.arange(len(labels))

bars = ax.barh(
    y,
    means,
    xerr=stds,
    capsize=6,
    color=COLORS,
    alpha=0.88,
    edgecolor="#1a1a1a",
    linewidth=1.0,
    height=0.65,
    zorder=3,
    error_kw={"elinewidth": 1.2, "capthick": 6, "alpha": 0.7},
)

# Value labels on bars
for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
    ax.text(
        m + s + 2.0,
        bar.get_y() + bar.get_height() / 2,
        f"{m:.1f}%",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="500",
        color="#1a1a1a",
    )

# Reference lines
if udr_sr:
    ax.axvline(
        udr_sr,
        color="#27AE60",
        ls="--",
        lw=2.2,
        alpha=0.85,
        label=f"Uniform DR (all params): {udr_sr:.1f}%",
        zorder=2,
    )
if naive_sr:
    ax.axvline(
        naive_sr,
        color="#E74C3C",
        ls=":",
        lw=2.0,
        alpha=0.80,
        label=f"No DR: {naive_sr:.1f}%",
        zorder=2,
    )
ax.axvline(
    pid_sr,
    color="#7F8C8D",
    ls="-.",
    lw=1.8,
    alpha=0.75,
    label=f"PID Baseline: {pid_sr:.0f}%",
    zorder=2,
)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10, color="#1a1a1a")
ax.set_xlabel(
    "Test Distribution Success Rate (%)", fontsize=11, fontweight="600", labelpad=10
)
ax.set_xlim(0, 115)
ax.set_ylim(-0.7, len(labels) - 0.3)
ax.set_title(
    "Ablation Study: Transfer Performance with Single Randomized Parameter",
    fontsize=12,
    fontweight="bold",
    loc="left",
    pad=15,
    color="#1a1a1a",
)
ax.legend(
    frameon=False, fontsize=9.5, loc="lower right", labelspacing=1.0, handlelength=2.5
)
ax.grid(axis="x", alpha=0.3, zorder=0, linestyle="-", linewidth=0.5, color="#d0d0d0")

# Annotation: which parameter matters most
if ablation:
    best = ablation[-1]  # highest success rate (sorted ascending)
    ax.annotate(
        f"Most impactful\nsingle parameter",
        xy=(best["sr_mean"], len(ablation) - 1),
        xytext=(best["sr_mean"] - 30, len(ablation) - 2.0),
        arrowprops=dict(
            arrowstyle="->",
            color="#2E86AB",
            lw=1.5,
            connectionstyle="arc3,rad=0.3",
        ),
        fontsize=8.5,
        color="#2E86AB",
        fontweight="600",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#2E86AB",
            linewidth=1.0,
            alpha=0.95,
        ),
    )

plt.tight_layout(pad=0.5)
out = FIG_DIR / "fig5_ablation.pdf"
plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
print(f"✓ Saved: {out}")
plt.close()

# Also save to Drive if on Colab
drive_path = Path("/content/drive/MyDrive/rl_research/auv/paper/figures")
if drive_path.exists():
    import shutil

    shutil.copy(out, drive_path / "fig5_ablation.pdf")
    print(f"✓ Copied to Drive: {drive_path / 'fig5_ablation.pdf'}")
