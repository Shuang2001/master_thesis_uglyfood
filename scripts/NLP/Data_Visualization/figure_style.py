# figure_style.py  –  shared palette & rcParams for all figures
# Import this at the top of every analysis script:
#   from figure_style import PALETTE, C, apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Muted academic palette (p3 hues, ~65% saturation) ──────────────────────
PALETTE = {
    "blue":   "#4A7FAF",   # primary
    "green":  "#5A9E5A",
    "orange": "#D4784E",
    "purple": "#8B72A8",
    "pink":   "#B8729A",
    "teal":   "#4AADBE",
    "grey":   "#8C8C8C",
    "light":  "#E8EEF4",   # background tint
    "dark":   "#2C3E50",   # text / axes
}

# Shorthand list for cycling
C = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
     PALETTE["purple"], PALETTE["pink"], PALETTE["teal"],
     PALETTE["grey"]]

def apply_style():
    """Call once per script to set global rcParams."""
    mpl.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     "#CCCCCC",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.color":         "#E5E5E5",
        "grid.linewidth":     0.6,
        "xtick.color":        "#555555",
        "ytick.color":        "#555555",
        "figure.facecolor":   "white",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })

def save_fig(path):
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
