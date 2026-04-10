"""
plot_transfer_results.py — Figure 5: A10a + A10c combined
==========================================================
Creates a single figure showing:
  Left panel:  Original test vs cross-model transfer (A10a)
  Right panel: Standard test vs extreme extrapolation (A10c)

Output: paper/figures/fig5_transfer_results.pdf
Usage:  python scripts/plot_transfer_results.py
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
MODES = ["none", "uniform", "curriculum"]
LABELS = {"none": "Naive SAC", "uniform": "Uniform DR", "curriculum": "CDR\n(Ours)"}
COLORS = ["#E24B4A", "#BA7517", "#1D9E75"]


def load_standard():
    """Load original paper results from energy_eval_results.json."""
    data = {}
    for mode in MODES:
        results = []
        for seed in [0, 1, 2]:
            for prefix in [mode, f"master_{mode}"]:
                p = BASE / prefix / f"{prefix}_seed{seed}" / "energy_eval_results.json"
                if p.exists():
                    with open(p) as f:
                        results.append(json.load(f))
                    break
        data[mode] = results
    return data


def load_transfer():
    """Load A10a cross-model transfer results."""
    p = BASE / "transfer_eval_a10a_results.json"
    if not p.exists():
        print(f"  WARNING: {p.name} not found — A10a not yet run")
        return None
    with open(p) as f:
        return json.load(f)


def load_extreme():
    """Load A10c extreme extrapolation results."""
    p = BASE / "extreme_eval_a10c_results.json"
    if not p.exists():
        print(f"  WARNING: {p.name} not found — A10c not yet run")
        return None
    with open(p) as f:
        return json.load(f)


def bar_group(
    ax,
    group_labels,
    group_data,
    group_colors,
    title,
    ylabel,
    show_pct=True,
    ylim=None,
    reference_line=None,
):
    """Helper: grouped bar chart."""
    n_groups = len(group_labels)
    x = np.arange(n_groups)
    bars = ax.bar(
        x,
        [d["mean"] * 100 if show_pct else d["mean"] for d in group_data],
        yerr=[d["std"] * 100 if show_pct else d["std"] for d in group_data],
        color=group_colors,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    for bar, d in zip(bars, group_data):
        val = d["mean"] * 100 if show_pct else d["mean"]
        std = d["std"] * 100 if show_pct else d["std"]
        fmt = f"{val:.1f}%" if show_pct else f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + 1.5,
            fmt,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if reference_line is not None:
        ax.axhline(
            y=reference_line,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label=f"PID ({reference_line:.0f}%)",
        )
        ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, pad=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, zorder=0)


def main():
    os.makedirs("paper/figures", exist_ok=True)

    std_data = load_standard()
    xfer_data = load_transfer()
    extr_data = load_extreme()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Panel A: Original test results ────────────────────────
    group_data = []
    for mode in MODES:
        results = std_data.get(mode, [])
        vals = [r["success_rate"] for r in results] if results else [0]
        group_data.append({"mean": np.mean(vals), "std": np.std(vals)})

    bar_group(
        axes[0],
        [LABELS[m] for m in MODES],
        group_data,
        COLORS,
        title="(a) Standard Transfer\n(held-out test dist)",
        ylabel="Success Rate (%)",
        show_pct=True,
        ylim=(0, 115),
        reference_line=3,
    )

    # ── Panel B: Cross-model transfer (A10a) ──────────────────
    if xfer_data:
        xfer_group = []
        for mode in MODES:
            results = xfer_data.get(mode, [])
            vals = [r["success_rate"] for r in results] if results else [0]
            xfer_group.append({"mean": np.mean(vals), "std": np.std(vals)})
        bar_group(
            axes[1],
            [LABELS[m] for m in MODES],
            xfer_group,
            COLORS,
            title="(b) Cross-Model Transfer\n(different AUV body, zero-shot)",
            ylabel="Success Rate (%)",
            show_pct=True,
            ylim=(0, 115),
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "A10a not yet run\n(run eval_transfer.py)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=10,
            color="gray",
        )
        axes[1].set_title("(b) Cross-Model Transfer\n(A10a pending)", fontsize=10)

    # ── Panel C: Extreme extrapolation (A10c) ─────────────────
    if extr_data:
        extreme_group = []
        for mode in MODES:
            results = extr_data.get("extreme", {}).get(mode, [])
            vals = [r["success_rate"] for r in results] if results else [0]
            extreme_group.append({"mean": np.mean(vals), "std": np.std(vals)})
        bar_group(
            axes[2],
            [LABELS[m] for m in MODES],
            extreme_group,
            COLORS,
            title="(c) Extreme Extrapolation\n(2× beyond training max)",
            ylabel="Success Rate (%)",
            show_pct=True,
            ylim=(0, 115),
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "A10c not yet run\n(run eval_extreme.py)",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            fontsize=10,
            color="gray",
        )
        axes[2].set_title("(c) Extreme Extrapolation\n(A10c pending)", fontsize=10)

    plt.suptitle(
        "Transfer Generalisation — Zero-Shot Robustness of CDR", fontsize=12, y=1.02
    )
    plt.tight_layout()

    out = "paper/figures/fig9_transfer_results.pdf"
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
