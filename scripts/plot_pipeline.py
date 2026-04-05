import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- 1. Dynamic Path Resolution (Repo-Safe) ---
# Resolves the absolute path of this script, then steps up one directory to the repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

# --- 2. Global Academic Typography Settings ---
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 42,  # TrueType fonts for IEEE/Academic compatibility
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.facecolor": "white",
    }
)

# --- 3. Cohesive Academic Color Palette ---
C_CORE_BG = "#E8F5EF"
C_CORE_FG = "#1D9E75"  # Soft Green
C_EXP_BG = "#EBF5FB"
C_EXP_FG = "#2980B9"  # Soft Blue
C_CON_BG = "#FDEDEC"
C_CON_FG = "#C0392B"  # Soft Red
C_SAC_BG = "#FEF5E7"
C_SAC_FG = "#D35400"  # Soft Orange
C_NEU_BG = "#F8F9F9"
C_NEU_FG = "#7F8C8D"  # Neutral Gray
C_DARK = "#2C3E50"
C_LINES = "#34495E"  # Text and Arrows


# --- 4. Drawing Helper Functions ---
def draw_node_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = None,
    bg_color: str = C_CORE_BG,
    fg_color: str = C_CORE_FG,
    is_bold: bool = True,
) -> None:
    """
    Draws a formatted rounded rectangle node for the flowchart.

    Args:
        ax (plt.Axes): The matplotlib axis to draw on.
        cx (float): Center X coordinate.
        cy (float): Center Y coordinate.
        width (float): Box width.
        height (float): Box height.
        title (str): Main text.
        subtitle (str, optional): Secondary subtext.
        bg_color (str): Background hex color.
        fg_color (str): Edge and text hex color.
        is_bold (bool): Whether the title should be bold.
    """
    box = mpatches.FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=fg_color,
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(box)

    title_weight = "bold" if is_bold else "normal"
    ax.text(
        cx,
        cy + (0.15 if subtitle else 0),
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight=title_weight,
        color=C_DARK,
        zorder=4,
    )

    if subtitle:
        ax.text(
            cx,
            cy - 0.20,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#444",
            zorder=4,
        )


def draw_routing_line(
    ax: plt.Axes,
    points: list,
    color: str = C_LINES,
    line_width: float = 1.5,
    line_style: str = "-",
) -> None:
    """Draws a connecting line segment between multiple coordinate tuples."""
    x_coords, y_coords = zip(*points)
    ax.plot(x_coords, y_coords, color=color, lw=line_width, ls=line_style, zorder=1)


def draw_arrow_head(
    ax: plt.Axes,
    start_pt: tuple,
    end_pt: tuple,
    color: str = C_LINES,
    line_width: float = 1.5,
    line_style: str = "-",
    label: str = None,
    label_pos: tuple = None,
) -> None:
    """Draws a directional arrow head, optionally attaching a text label."""
    ax.annotate(
        "",
        xy=end_pt,
        xytext=start_pt,
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.4,head_length=0.6",
            color=color,
            lw=line_width,
            ls=line_style,
        ),
        zorder=2,
    )
    if label and label_pos:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=2),
            zorder=3,
        )


# --- 5. Main Diagram Generation ---
def generate_cdr_pipeline_diagram(output_dir: Path) -> None:
    """
    Constructs the CDR algorithm flowchart and saves it to disk.

    Args:
        output_dir (Path): The directory where the PDF/PNG files will be saved.
    """
    # Early Return: Ensure output directory exists before doing heavy plotting
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box_w, box_h = 2.4, 0.9

    # Backbone
    draw_node_box(
        ax,
        1.6,
        3.5,
        box_w,
        box_h,
        "Episode Reset",
        r"Sample $\phi \sim \mathcal{U}(r)$",
        bg_color=C_NEU_BG,
        fg_color=C_NEU_FG,
    )
    draw_node_box(
        ax, 4.8, 3.5, box_w, box_h, "Run Episode", r"$500$ steps $\cdot$ $25$Hz"
    )
    draw_node_box(ax, 8.0, 3.5, box_w, box_h, "Update Window", r"$W=50$ episodes")
    draw_node_box(
        ax, 11.2, 3.5, box_w, box_h, "Check Success Rate", r"$sr = \mu_{window}$"
    )

    # Branches
    draw_node_box(
        ax,
        14.0,
        5.0,
        box_w,
        box_h,
        "EXPAND (+5%)",
        r"$r \leftarrow r \cdot 1.05$",
        bg_color=C_EXP_BG,
        fg_color=C_EXP_FG,
    )
    draw_node_box(
        ax,
        14.0,
        2.0,
        box_w,
        box_h,
        "CONTRACT (-3%)",
        r"$r \leftarrow r \cdot 0.97$",
        bg_color=C_CON_BG,
        fg_color=C_CON_FG,
    )

    # Aux Nodes
    draw_node_box(
        ax,
        4.8,
        1.4,
        2.0,
        0.7,
        "SAC Update",
        r"Actor & Critic",
        bg_color=C_SAC_BG,
        fg_color=C_SAC_FG,
    )
    draw_node_box(
        ax,
        4.8,
        5.8,
        2.0,
        0.7,
        r"Track $\lambda \in [0,1]$",
        r"Curriculum level",
        bg_color=C_NEU_BG,
        fg_color=C_NEU_FG,
    )

    # Backbone Flow
    draw_arrow_head(ax, (2.8, 3.5), (3.6, 3.5))
    draw_arrow_head(ax, (6.0, 3.5), (6.8, 3.5))
    draw_arrow_head(ax, (9.2, 3.5), (10.0, 3.5))

    # Branching Up (Expand)
    draw_routing_line(ax, [(12.4, 3.5), (12.8, 3.5), (12.8, 5.0)])
    draw_arrow_head(
        ax,
        (12.8, 5.0),
        (12.8, 5.0),
        color=C_EXP_FG,
        label=r"$sr > \tau_{upper}$",
        label_pos=(12.8, 4.3),
    )
    draw_arrow_head(ax, (12.8, 5.0), (13.2, 5.0), color=C_EXP_FG)

    # Branching Down (Contract)
    draw_routing_line(ax, [(12.4, 3.5), (12.8, 3.5), (12.8, 2.0)])
    draw_arrow_head(
        ax,
        (12.8, 2.0),
        (13.2, 2.0),
        color=C_CON_FG,
        label=r"$sr < \tau_{lower}$",
        label_pos=(12.8, 2.7),
    )

    # Maintain (Pass-through)
    draw_routing_line(ax, [(12.4, 3.5), (15.5, 3.5)], line_style="--", color=C_NEU_FG)
    ax.text(
        14.0,
        3.75,
        r"$\tau_{lower} \le sr \le \tau_{upper}$",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_NEU_FG,
        backgroundcolor="white",
        zorder=3,
    )

    # SAC Flow
    draw_arrow_head(ax, (4.8, 3.05), (4.8, 1.75), color=C_SAC_FG)

    # Tracking Feedback
    draw_routing_line(
        ax, [(14.0, 5.45), (14.0, 6.3), (4.8, 6.3)], line_style="--", color=C_NEU_FG
    )
    draw_arrow_head(ax, (4.8, 6.3), (4.8, 6.15), line_style="--", color=C_NEU_FG)

    # The Loop Back
    draw_routing_line(
        ax,
        [(15.2, 5.0), (15.5, 5.0), (15.5, 0.6), (1.6, 0.6)],
        line_style="--",
        color=C_NEU_FG,
    )
    draw_routing_line(ax, [(15.2, 2.0), (15.5, 2.0)], line_style="--", color=C_NEU_FG)
    draw_arrow_head(ax, (1.6, 0.6), (1.6, 3.05), line_style="--", color=C_NEU_FG)

    ax.text(
        8.0,
        0.85,
        r"Next episode $\rightarrow$ Repeat training loop",
        ha="center",
        va="center",
        fontsize=11,
        color=C_NEU_FG,
        style="italic",
        backgroundcolor="white",
        zorder=3,
    )

    # Title & Legend
    ax.set_title(
        "Curriculum Domain Randomisation (CDR) — Algorithm Flow",
        fontsize=14,
        fontweight="bold",
        pad=20,
        color=C_DARK,
    )

    legend_items = [
        mpatches.Patch(
            facecolor=C_EXP_BG, edgecolor=C_EXP_FG, label="Expand bounds (Harder)"
        ),
        mpatches.Patch(
            facecolor=C_CON_BG, edgecolor=C_CON_FG, label="Contract bounds (Easier)"
        ),
        mpatches.Patch(
            facecolor=C_SAC_BG, edgecolor=C_SAC_FG, label="Policy/Value Gradients"
        ),
        mpatches.Patch(
            facecolor=C_CORE_BG, edgecolor=C_CORE_FG, label="CDR Core State"
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()

    pdf_out = output_dir / "fig1_pipeline.pdf"
    png_out = output_dir / "fig1_pipeline.png"

    plt.savefig(str(pdf_out), dpi=400, bbox_inches="tight")
    plt.savefig(str(png_out), dpi=400, bbox_inches="tight")

    logging.info(f"✓ PDF saved for LaTeX: {pdf_out}")
    logging.info(f"✓ PNG saved for Web: {png_out}")

    # Close the figure to free up memory (best practice for scripts)
    plt.close(fig)


if __name__ == "__main__":
    generate_cdr_pipeline_diagram(OUTPUT_DIR)
