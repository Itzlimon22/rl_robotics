cat > ~/rl_robotics/scripts/plot_pipeline.py << 'PYEOF'
"""
plot_pipeline.py — Figure 1: CDR algorithm flow diagram
Output: paper/figures/fig1_pipeline.pdf
Usage:  python scripts/plot_pipeline.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

mpl.rcParams.update({
    "font.family":"serif","font.size":10,
    "pdf.fonttype":42,"savefig.dpi":300,
})
os.makedirs("paper/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")

C_GREEN = "#1D9E75"; C_LIGHT = "#E8F5EF"; C_GRAY = "#F5F5F5"
C_BLUE  = "#3498DB"; C_RED   = "#E74C3C"; C_DARK = "#2C3E50"
C_ORG   = "#E67E22"; C_ORG_L = "#FFF3CD"

def box(ax, x, y, w, h, txt, sub=None, fc=C_LIGHT, ec=C_GREEN, fs=9, bold=False):
    r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.06", facecolor=fc, edgecolor=ec, lw=1.5, zorder=3)
    ax.add_patch(r)
    ax.text(x, y+(0.13 if sub else 0), txt,
            ha="center", va="center", fontsize=fs, color=C_DARK,
            fontweight="bold" if bold else "normal", zorder=4)
    if sub:
        ax.text(x, y-0.22, sub, ha="center", va="center",
                fontsize=7.5, color="#555", style="italic", zorder=4)

def arr(ax, x1, y1, x2, y2, c="#555", lbl=None):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="->", color=c, lw=1.5), zorder=5)
    if lbl:
        ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.15, lbl,
                ha="center", fontsize=7.5, color=c, style="italic")

# Main flow
box(ax, 1.4, 2.7, 2.2, 0.8, "Episode Reset", "Sample φ~Uniform(ranges)", fc=C_GRAY, ec="#888")
box(ax, 3.8, 2.7, 1.8, 0.8, "Run Episode",   "500 steps · 25Hz")
box(ax, 5.9, 2.7, 2.0, 0.8, "Update Window", "W=50 episodes rolling")
box(ax, 8.1, 2.7, 2.0, 0.8, "Check sr",      "sr = mean(window)")

arr(ax, 2.5, 2.7, 2.9, 2.7)
arr(ax, 4.7, 2.7, 4.9, 2.7)
arr(ax, 6.9, 2.7, 7.1, 2.7)

# Expand / Contract
box(ax, 10.2, 4.2, 2.2, 0.7, "EXPAND (+5%)", "ranges → wider",
    fc="#D5EEF8", ec=C_BLUE, bold=True)
box(ax, 10.2, 1.2, 2.2, 0.7, "CONTRACT (−3%)", "ranges → narrower",
    fc="#FAE5E5", ec=C_RED, bold=True)

arr(ax, 9.1, 3.1, 10.2, 3.85, c=C_BLUE, lbl="sr > 0.70")
arr(ax, 9.1, 2.3, 10.2, 1.55, c=C_RED,  lbl="sr < 0.40")

# SAC Update
box(ax, 5.9, 1.0, 1.8, 0.7, "SAC Update", "Actor + Critic", fc=C_ORG_L, ec=C_ORG)
arr(ax, 5.9, 2.3, 5.9, 1.35, c=C_ORG)

# Curriculum level tracker
box(ax, 3.5, 4.4, 2.0, 0.65, "Track λ ∈ [0,1]", "Curriculum level", fc="#EEE", ec="#999")
ax.plot([10.2,10.2,3.5,3.5],[3.85,4.4,4.4,4.4], color="#999", lw=1.0, ls="--", zorder=2)
arr(ax, 3.5, 4.08, 3.5, 3.08, c="#999")

# Loop arrow
ax.annotate("", xy=(1.4, 2.3), xytext=(1.4, 0.4),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.plot([1.4, 11.0, 11.0, 1.4], [0.4, 0.4, 2.7, 2.7],
        color="#555", lw=1.0, ls=":", zorder=2)
ax.text(6.2, 0.18, "Next episode → repeat", ha="center", fontsize=8,
        color="#555", style="italic")

ax.set_title("Curriculum Domain Randomisation (CDR) — Training Loop",
             fontsize=12, fontweight="bold", pad=12, color=C_DARK)

legend_items = [
    mpatches.Patch(facecolor=C_LIGHT, edgecolor=C_GREEN, label="CDR mechanism"),
    mpatches.Patch(facecolor="#D5EEF8", edgecolor=C_BLUE, label="Expand (sr > 0.70)"),
    mpatches.Patch(facecolor="#FAE5E5", edgecolor=C_RED,  label="Contract (sr < 0.40)"),
    mpatches.Patch(facecolor=C_ORG_L,  edgecolor=C_ORG,  label="SAC gradient update"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout()

out = "paper/figures/fig1_pipeline.pdf"
plt.savefig(out, bbox_inches="tight"); print(f"✓ {out}"); plt.close()
PYEOF

cd ~/rl_robotics && python scripts/plot_pipeline.py