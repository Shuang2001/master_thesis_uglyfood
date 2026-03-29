import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

COL_STAGE   = "#5B8DB8"   # 蓝
COL_FINAL   = "#6AAF6A"   # 绿
COL_EXCLUDE = "#E8956D"   # 橙
WHITE       = "white"
DARK        = "#2C2C2C"
GREY        = "#999999"

# ── Layout constants ───────────────────────────────────────────────────────
# Main column x-centre, exclusion column x-centre
MX   = 0.32   # main boxes centre-x  (shifted right to give stage labels room)
EX   = 0.75   # exclusion boxes centre-x
MW   = 0.36   # main box width
EW   = 0.30   # exclusion box width
MH   = 0.11   # main box height
EH   = 0.09   # exclusion box height

# y positions for main boxes (top → bottom)
MY = [0.88, 0.69, 0.50, 0.31, 0.10]
# y positions for exclusion boxes
EY = [0.785, 0.595, 0.405, 0.205]

STAGES = [
    ("Records identified\nvia Scopus database\nsearching",       "n = 2,285"),
    ("Records after\nduplicate removal",                          "n = 2,285"),
    ("Records after\nkeyword pre-filtering\n(title + abstract)",  "n = 1,140"),
    ("Records after\nprecision filtering\n(consumer + purchase terms)", "n = 169"),
    ("Studies included\nin final corpus\n(ML ranking + manual review)", "n = 84"),
]
STAGE_COLS = [COL_STAGE, COL_STAGE, COL_STAGE, COL_STAGE, COL_FINAL]

EXCLUSIONS = [
    ("Duplicates removed",                          "n = 0"),
    ("Excluded: off-topic\n(keyword mismatch)",     "n = 1,145"),
    ("Excluded: no consumer\nor purchase focus",    "n = 971"),
    ("Excluded: low ML relevance\nscore / manual rejection", "n = 85"),
]

STAGE_LABELS = ["Identification", "Deduplication", "Screening", "Eligibility", "Included"]

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def rounded_box(cx, cy, w, h, color, label, n, fontsize=11):
    x, y = cx - w/2, cy - h/2
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.012",
                         facecolor=color, edgecolor=WHITE,
                         linewidth=1.5, zorder=3,
                         transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(cx, cy + 0.016, label,
            ha="center", va="center", fontsize=fontsize,
            color=WHITE, fontweight="bold",
            transform=ax.transAxes, zorder=4, linespacing=1.4)
    ax.text(cx, cy - 0.030, n,
            ha="center", va="center", fontsize=fontsize + 1,
            color=WHITE, fontweight="bold",
            transform=ax.transAxes, zorder=4)

# ── Stage labels: placed to the LEFT of main boxes, never overlapping ─────
# Put them at x = MX - MW/2 - 0.02 (just outside left edge of boxes)
LABEL_X = MX - MW/2 - 0.03   # right-aligned to this x
for i, sl in enumerate(STAGE_LABELS):
    ax.text(LABEL_X, MY[i],
            sl,
            ha="right", va="center", fontsize=10,
            color=GREY, style="italic",
            transform=ax.transAxes)

# ── Main boxes ─────────────────────────────────────────────────────────────
for i, ((lbl, n), col) in enumerate(zip(STAGES, STAGE_COLS)):
    rounded_box(MX, MY[i], MW, MH, col, lbl, n)

# ── Downward arrows between main boxes ────────────────────────────────────
for i in range(len(MY) - 1):
    y0 = MY[i]   - MH/2
    y1 = MY[i+1] + MH/2
    ax.annotate("", xy=(MX, y1 + 0.003),
                xytext=(MX, y0 - 0.003),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8),
                zorder=2)

# ── Exclusion boxes ────────────────────────────────────────────────────────
for lbl, n, ey in zip([e[0] for e in EXCLUSIONS],
                      [e[1] for e in EXCLUSIONS],
                      EY):
    rounded_box(EX, ey, EW, EH, COL_EXCLUDE, lbl, n, fontsize=10.5)

# ── Side arrows: right edge of main → left edge of exclusion ──────────────
connect_pairs = [(0, 0), (2, 1), (3, 2), (4, 3)]
for si, ei in connect_pairs:
    sx, sy = MX, MY[si]
    ey_    = EY[ei]
    ax.annotate("",
                xy=(EX - EW/2 - 0.005, ey_),
                xytext=(MX + MW/2 + 0.005, sy),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",
                                color=COL_EXCLUDE,
                                lw=1.5,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)

# ── Legend ─────────────────────────────────────────────────────────────────
leg = [mpatches.Patch(facecolor=COL_STAGE,   label="Screening stage"),
       mpatches.Patch(facecolor=COL_FINAL,   label="Final included studies"),
       mpatches.Patch(facecolor=COL_EXCLUDE, label="Excluded records")]
ax.legend(handles=leg, loc="lower right",
          bbox_to_anchor=(0.98, 0.01),
          frameon=True, fontsize=10,
          edgecolor="#CCCCCC")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure1_fixed.png",
            dpi=200, bbox_inches="tight")
print("saved")
