%%bash
mkdir -p /content/rl_robotics/scripts
cat > /content/rl_robotics/scripts/plot_pipeline.py << 'PYEOF'
"""
plot_pipeline.py — Figure 1: CDR algorithm flow diagram (Paper Ready)
"""
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Global Academic Typography Settings ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "pdf.fonttype": 42, # TrueType fonts for IEEE/Academic compatibility
    "ps.fonttype": 42,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.facecolor": "white"
})

# --- 2. Cohesive Academic Color Palette ---
C_CORE_BG = "#E8F5EF"; C_CORE_FG = "#1D9E75" # Soft Green (CDR backbone)
C_EXP_BG  = "#EBF5FB"; C_EXP_FG  = "#2980B9" # Soft Blue (Expand)
C_CON_BG  = "#FDEDEC"; C_CON_FG  = "#C0392B" # Soft Red (Contract)
C_SAC_BG  = "#FEF5E7"; C_SAC_FG  = "#D35400" # Soft Orange (SAC)
C_NEU_BG  = "#F8F9F9"; C_NEU_FG  = "#7F8C8D" # Neutral Gray (Track/Reset)
C_DARK    = "#2C3E50"; C_LINES   = "#34495E" # Text and Arrows

# --- 3. Drawing Helper Functions ---
def draw_box(ax, cx, cy, w, h, title, subtitle=None, bg=C_CORE_BG, fg=C_CORE_FG, bold_title=True):
    """Draws a rounded rectangle node for the flowchart."""
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=bg, edgecolor=fg, linewidth=1.8, zorder=3
    )
    ax.add_patch(box)
    
    title_weight = "bold" if bold_title else "normal"
    ax.text(cx, cy + (0.15 if subtitle else 0), title,
            ha="center", va="center", fontsize=11, fontweight=title_weight, color=C_DARK, zorder=4)
    if subtitle:
        ax.text(cx, cy - 0.20, subtitle,
                ha="center", va="center", fontsize=9.5, color="#444", zorder=4)

def draw_line(ax, points, color=C_LINES, lw=1.5, ls="-"):
    """Draws a line segment between multiple coordinates."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1)

def draw_arrow(ax, pt1, pt2, color=C_LINES, lw=1.5, ls="-", label=None, label_pos=None):
    """Draws the final arrow head segment, optionally with a label."""
    ax.annotate("", xy=pt2, xytext=pt1,
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", color=color, lw=lw, ls=ls), zorder=2)
    if label and label_pos:
        ax.text(label_pos[0], label_pos[1], label, ha="center", va="center",
                fontsize=10, color=color,
                bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=3)

# --- 4. Main Diagram Generation ---
def generate_diagram(output_path_png: Path, output_path_pdf: Path):
    # Setup canvas (14x7 provides clean breathing room for 5 columns of nodes)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off") # Hide coordinate axes

    # --- Node Placement (Centers) ---
    W, H = 2.4, 0.9 # Standard Width/Height
    
    # Backbone
    draw_box(ax, 1.6, 3.5, W, H, "Episode Reset", r"Sample $\phi \sim \mathcal{U}(r)$", bg=C_NEU_BG, fg=C_NEU_FG)
    draw_box(ax, 4.8, 3.5, W, H, "Run Episode", r"$500$ steps $\cdot$ $25$Hz")
    draw_box(ax, 8.0, 3.5, W, H, "Update Window", r"$W=50$ episodes")
    draw_box(ax, 11.2, 3.5, W, H, "Check Success Rate", r"$sr = \mu_{window}$")
    
    # Branches
    draw_box(ax, 14.0, 5.0, W, H, "EXPAND (+5%)", r"$r \leftarrow r \cdot 1.05$", bg=C_EXP_BG, fg=C_EXP_FG)
    draw_box(ax, 14.0, 2.0, W, H, "CONTRACT (-3%)", r"$r \leftarrow r \cdot 0.97$", bg=C_CON_BG, fg=C_CON_FG)
    
    # Aux Nodes
    draw_box(ax, 4.8, 1.4, 2.0, 0.7, "SAC Update", r"Actor & Critic", bg=C_SAC_BG, fg=C_SAC_FG)
    draw_box(ax, 4.8, 5.8, 2.0, 0.7, r"Track $\lambda \in [0,1]$", r"Curriculum level", bg=C_NEU_BG, fg=C_NEU_FG)

    # --- Orthogonal Arrow Routing ---
    # Backbone Flow
    draw_arrow(ax, (2.8, 3.5), (3.6, 3.5))
    draw_arrow(ax, (6.0, 3.5), (6.8, 3.5))
    draw_arrow(ax, (9.2, 3.5), (10.0, 3.5))

    # Branching Up (Expand)
    draw_line(ax, [(12.4, 3.5), (12.8, 3.5), (12.8, 5.0)])
    draw_arrow(ax, (12.8, 5.0), (12.8, 5.0), color=C_EXP_FG, label=r"$sr > \tau_{upper}$", label_pos=(12.8, 4.3)) # Using 0-length arrow to just draw head over line
    draw_arrow(ax, (12.8, 5.0), (13.2, 5.0), color=C_EXP_FG)

    # Branching Down (Contract)
    draw_line(ax, [(12.4, 3.5), (12.8, 3.5), (12.8, 2.0)])
    draw_arrow(ax, (12.8, 2.0), (13.2, 2.0), color=C_CON_FG, label=r"$sr < \tau_{lower}$", label_pos=(12.8, 2.7))

    # Maintain (Pass-through)
    draw_line(ax, [(12.4, 3.5), (15.5, 3.5)], ls="--", color=C_NEU_FG)
    ax.text(14.0, 3.75, r"$\tau_{lower} \le sr \le \tau_{upper}$", ha="center", va="center", 
            fontsize=9.5, color=C_NEU_FG, backgroundcolor="white", zorder=3)

    # SAC Flow
    draw_arrow(ax, (4.8, 3.05), (4.8, 1.75), color=C_SAC_FG)

    # Tracking Feedback
    draw_line(ax, [(14.0, 5.45), (14.0, 6.3), (4.8, 6.3)], ls="--", color=C_NEU_FG)
    draw_arrow(ax, (4.8, 6.3), (4.8, 6.15), ls="--", color=C_NEU_FG)

    # The Loop Back (Next Episode Backbone)
    draw_line(ax, [(15.2, 5.0), (15.5, 5.0), (15.5, 0.6), (1.6, 0.6)], ls="--", color=C_NEU_FG)
    draw_line(ax, [(15.2, 2.0), (15.5, 2.0)], ls="--", color=C_NEU_FG)
    draw_arrow(ax, (1.6, 0.6), (1.6, 3.05), ls="--", color=C_NEU_FG)
    
    ax.text(8.0, 0.85, r"Next episode $\rightarrow$ Repeat training loop", ha="center", va="center",
            fontsize=11, color=C_NEU_FG, style="italic", backgroundcolor="white", zorder=3)

    # --- Title & Legend ---
    ax.set_title("Curriculum Domain Randomisation (CDR) — Algorithm Flow",
                 fontsize=14, fontweight="bold", pad=20, color=C_DARK)

    legend_items = [
        mpatches.Patch(facecolor=C_EXP_BG,  edgecolor=C_EXP_FG,  label="Expand bounds (Harder)"),
        mpatches.Patch(facecolor=C_CON_BG,  edgecolor=C_CON_FG,  label="Contract bounds (Easier)"),
        mpatches.Patch(facecolor=C_SAC_BG,  edgecolor=C_SAC_FG,  label="Policy/Value Gradients"),
        mpatches.Patch(facecolor=C_CORE_BG, edgecolor=C_CORE_FG, label="CDR Core State"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.05), 
              ncol=4, fontsize=10, frameon=False)

    # --- Save Execution ---
    plt.tight_layout()
    
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PDF for LaTeX inclusion (Scalable Vector)
    plt.savefig(str(output_path_pdf), dpi=400, bbox_inches="tight")
    # Save PNG for quick visualization / Web
    plt.savefig(str(output_path_png), dpi=400, bbox_inches="tight")
    
    logging.info(f"✓ Publication PDF saved: {output_path_pdf}")
    logging.info(f"✓ Preview PNG saved: {output_path_png}")
    plt.show()

if __name__ == "__main__":
    # FIX: Environment-Aware Drive Mount
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        BASE_DIR = Path('/content/drive/MyDrive/rl_research/auv/paper/figures')
        logging.info("Google Drive explicitly mounted.")
    except ImportError:
        BASE_DIR = Path.home() / "rl_research" / "auv" / "paper" / "figures"
        logging.info("Local environment detected. Saving to home directory.")

    pdf_out = BASE_DIR / "fig1_pipeline.pdf"
    png_out = BASE_DIR / "fig1_pipeline.png"
    
    generate_diagram(png_out, pdf_out)
PYEOF

python3 /content/rl_robotics/scripts/plot_pipeline.py