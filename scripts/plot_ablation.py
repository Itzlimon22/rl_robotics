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
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
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

# Color: ablation bars in blue, baselines as reference lines
COLORS = ["#3498DB"] * len(ablation)

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(labels))

bars = ax.barh(
    y,
    means,
    xerr=stds,
    capsize=5,
    color=COLORS,
    alpha=0.85,
    edgecolor="white",
    height=0.6,
    zorder=3,
)

# Value labels
for bar, m, s in zip(bars, means, stds):
    ax.text(
        m + s + 1.5,
        bar.get_y() + bar.get_height() / 2,
        f"{m:.1f}%",
        va="center",
        fontsize=9,
        color="#333",
    )

# Reference lines
if udr_sr:
    ax.axvline(
        udr_sr,
        color="#1D9E75",
        ls="--",
        lw=2.0,
        alpha=0.8,
        label=f"Uniform DR (all params): {udr_sr:.1f}%",
    )
if naive_sr:
    ax.axvline(
        naive_sr,
        color="#E24B4A",
        ls=":",
        lw=1.5,
        alpha=0.8,
        label=f"No DR: {naive_sr:.1f}%",
    )
ax.axvline(
    pid_sr, color="#555555", ls="-.", lw=1.5, alpha=0.7, label=f"PID: {pid_sr:.0f}%"
)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Test Distribution Success Rate (%)", fontsize=12)
ax.set_xlim(0, 115)
ax.set_title(
    "Ablation: Transfer Success When Only One Parameter Is Randomised",
    fontsize=11,
    fontweight="bold",
    loc="left",
)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25, zorder=0)

# Annotation: which parameter matters most
if ablation:
    best = ablation[-1]  # highest success rate (sorted ascending)
    ax.annotate(
        f"← Most impactful\n  single parameter",
        xy=(best["sr_mean"], len(ablation) - 1),
        xytext=(best["sr_mean"] - 25, len(ablation) - 1.4),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
        fontsize=8,
        color="#333",
    )

plt.tight_layout()
out = FIG_DIR / "fig5_ablation.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"✓ Saved: {out}")
plt.close()

# Also save to Drive if on Colab
drive_path = Path("/content/drive/MyDrive/rl_research/auv/paper/figures")
if drive_path.exists():
    import shutil

    shutil.copy(out, drive_path / "fig5_ablation.pdf")
    print(f"✓ Copied to Drive: {drive_path / 'fig5_ablation.pdf'}")
