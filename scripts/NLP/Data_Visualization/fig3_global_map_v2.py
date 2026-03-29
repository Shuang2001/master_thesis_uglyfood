# fig3_global_map_v2.py  –  Research hotspot world map  (no title, overlap fixed)
# ============================================================
# Prerequisites: unzip ne_110m_admin_0_countries.zip into the same folder.
# Required files:
#   ne_110m_admin_0_countries.shp
#   ne_110m_admin_0_countries.dbf
#   ne_110m_admin_0_countries.shx
#
# Install:  pip install geopandas matplotlib pandas
# Run:      python fig3_global_map_v2.py
# Output:   data_processed/Figure3_global_map_v2.png
# ============================================================

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import PALETTE, apply_style, save_fig

apply_style()

OUT  = "data_processed/Figure3_global_map_v2.png"
DATA = "data_processed/bertopic_doc_topics.csv"
SHP  = "ne_110m_admin_0_countries.shp"

if not os.path.exists(SHP):
    raise FileNotFoundError(
        f"\nCannot find {SHP}\n"
        "Place the unzipped Natural Earth files in the thesis_figures/ directory.\n"
        "Required: .shp  .dbf  .shx"
    )

# ── Country publication counts ────────────────────────────────────────────
df = pd.read_csv(DATA, encoding="cp1252", on_bad_lines="skip")
counter = Counter()
for val in df["Country"].dropna():
    for c in str(val).split(";"):
        c = c.strip()
        if c:
            counter[c] += 1

# ── CSV name → Natural Earth NAME mapping ────────────────────────────────
NAME_MAP = {
    "United States":  "United States of America",
    "United Kingdom": "United Kingdom",
    "South Korea":    "Republic of Korea",
    "Czech Republic": "Czech Republic",
    "Taiwan":         "Taiwan",
    "Netherlands":    "Netherlands",
    "Switzerland":    "Switzerland",
    "Norway":         "Norway",
    "Denmark":        "Denmark",
    "Sweden":         "Sweden",
    "Germany":        "Germany",
    "France":         "France",
    "Brazil":         "Brazil",
    "Australia":      "Australia",
    "Thailand":       "Thailand",
    "Belgium":        "Belgium",
    "Poland":         "Poland",
    "Portugal":       "Portugal",
    "Israel":         "Israel",
    "Canada":         "Canada",
    "Malaysia":       "Malaysia",
    "New Zealand":    "New Zealand",
    "Italy":          "Italy",
    "India":          "India",
    "Spain":          "Spain",
    "Indonesia":      "Indonesia",
    "China":          "China",
}

# ── Load shapefile ────────────────────────────────────────────────────────
world = gpd.read_file(SHP)

name_col = None
for candidate in ["NAME", "ADMIN", "NAME_LONG", "SOVEREIGNT", "name"]:
    if candidate in world.columns:
        name_col = candidate
        break

if name_col is None:
    raise ValueError("Cannot find country name column. Check the printed column list.")

world = world.rename(columns={name_col: "name"})
mapped = {NAME_MAP.get(k, k): v for k, v in counter.items()}
world["pub_count"] = world["name"].map(mapped).fillna(0).astype(int)

# ── Bubble coordinates (lon, lat) ────────────────────────────────────────
COORDS = {
    "United States of America": (-98,  38),
    "China":                    (105,  35),
    "Germany":                  ( 10,  51),
    "Netherlands":              (  5,  52.5),
    "Denmark":                  (  9,  56),
    "United Kingdom":           ( -2,  54),
    "Switzerland":              (  8,  47),
    "France":                   (  2,  46),
    "Taiwan":                   (121,  23.5),
    "Sweden":                   ( 18,  60),
    "Brazil":                   (-51, -14),
    "Norway":                   ( 10,  62),
    "Thailand":                 (101,  15),
    "Australia":                (134, -25),
    "Czech Republic":           ( 16,  50),
    "Portugal":                 ( -8,  39),
    "Israel":                   ( 35,  31),
    "Belgium":                  (  4,  50.5),
    "Canada":                   (-96,  56),
    "Malaysia":                 (110,   3),
    "New Zealand":              (172, -42),
    "Republic of Korea":        (128,  37),
    "Poland":                   ( 20,  52),
    "Italy":                    ( 12,  43),
    "India":                    ( 79,  22),
    "Spain":                    ( -3,  40),
    "Indonesia":                (114,  -0.5),
}

def bubble_color(n):
    if n >= 12: return PALETTE["orange"]
    if n >= 6:  return PALETTE["teal"]
    return PALETTE["blue"]

max_count = max(counter.values())

# ── Figure setup ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
ax.set_facecolor("#C8DCF0")

# 1) Base map (all countries)
world.plot(
    ax=ax,
    color="#DDE6ED",
    edgecolor="#A8B8C2",
    linewidth=0.35,
    zorder=1,
)

# 2) Countries with publications — choropleth overlay
world[world["pub_count"] > 0].plot(
    ax=ax,
    column="pub_count",
    cmap="Blues",
    vmin=0, vmax=max_count,
    edgecolor="#7A9CB0",
    linewidth=0.5,
    alpha=0.45,
    zorder=2,
)

# 3) Bubbles (draw smallest first so largest sits on top)
plot_items = [(NAME_MAP.get(c, c), c, n) for c, n in counter.items() if n > 0]
plot_items.sort(key=lambda x: x[2])

for ne_name, orig_name, count in plot_items:
    if ne_name not in COORDS:
        continue
    lon, lat = COORDS[ne_name]
    size = 60 + (count / max_count) ** 0.55 * 1800
    col  = bubble_color(count)

    ax.scatter(lon, lat, s=size * 1.9, color=col,
               alpha=0.12, zorder=3, linewidths=0)
    ax.scatter(lon, lat, s=size, color=col,
               edgecolors="white", linewidth=1.8,
               alpha=0.90, zorder=4)
    # Skip number labels for crowded European countries —
    # their counts are already shown in the callout panel on the left.
    EUROPE_SKIP = {
        "Germany", "Netherlands", "United Kingdom", "Switzerland",
        "France", "Czech Republic", "Belgium", "Poland",
        "Portugal", "Italy", "Spain", "Norway", "Sweden", "Denmark",
    }
    if size > 250 and ne_name not in EUROPE_SKIP:
        ax.text(lon, lat, str(count),
                ha="center", va="center",
                fontsize=9 if count < 10 else 10,
                fontweight="bold", color="white", zorder=5)

# 4) Direct labels for non-European countries (≥ 2 publications)
DIRECT_OFFSETS = {
    # Moved left and slightly lower so the label box clears Canada's bubble
    "United States of America": (-22, 6),
    "China":                    (  0, 10),
    "Taiwan":                   ( 14, -6),
    "Brazil":                   (-20,  0),  # moved left, away from Europe callout
    "Australia":                (  0,-22),  # further below bubble
    "Norway":                   (-16,  7),
    "Sweden":                   ( 18,  7),
    "Denmark":                  ( 20,  3),
    "Thailand":                 ( 16, -6),
    # Moved further right so it doesn't conflict with the US label
    "Canada":                   ( 20, 10),
}

EUROPE_NE = {
    "Germany", "Netherlands", "United Kingdom", "Switzerland",
    "France", "Czech Republic", "Belgium", "Poland",
    "Portugal", "Italy", "Spain"
}

for orig_name, count in counter.items():
    if count < 2:
        continue
    ne_name = NAME_MAP.get(orig_name, orig_name)
    if ne_name not in COORDS or ne_name in EUROPE_NE:
        continue
    lon, lat = COORDS[ne_name]
    dx, dy   = DIRECT_OFFSETS.get(ne_name, (0, 9))
    col      = bubble_color(count)

    ax.annotate(
        f"{orig_name}\nn = {count}",
        xy=(lon, lat), xytext=(lon + dx, lat + dy),
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=col, linewidth=1.0, alpha=0.93),
        arrowprops=dict(arrowstyle="-", color=col, lw=0.9)
                  if (abs(dx) + abs(dy)) > 3 else None,
        zorder=7,
    )

# 5) European callout panel — left side, evenly spaced
#
#    Key changes vs v1:
#    - Callout anchor moved further left (-38) and wider spacing (8.5 pt)
#      so labels never crowd each other.
#    - Countries are sorted by publication count (descending) so the most
#      prominent entries sit at the top.
#    - arc3 radius reduced to 0.05 for tidier connecting lines.

europe_order = [
    ("Germany",        "Germany"),
    ("Netherlands",    "Netherlands"),
    ("United Kingdom", "United Kingdom"),
    ("Switzerland",    "Switzerland"),
    ("France",         "France"),
    ("Czech Republic", "Czech Republic"),
    ("Belgium",        "Belgium"),
    ("Poland",         "Poland"),
    ("Portugal",       "Portugal"),
    ("Italy",          "Italy"),
    ("Spain",          "Spain"),
]
europe_entries = [
    (orig, ne, counter.get(orig, 0))
    for orig, ne in europe_order
    if counter.get(orig, 0) > 0 and ne in COORDS
]
europe_entries.sort(key=lambda x: -x[2])   # highest count first

# Callout column position
CALLOUT_X     = -38   # text anchor x (right edge of label box)
CALLOUT_Y_TOP =  70   # top label y
LINE_GAP      =   8.5 # vertical gap between labels (was 7.0 — more breathing room)

for i, (orig, ne, count) in enumerate(europe_entries):
    lon, lat = COORDS[ne]
    col      = bubble_color(count)
    lx       = CALLOUT_X + 11
    ly       = CALLOUT_Y_TOP - i * LINE_GAP

    # Connecting line from label to bubble
    ax.annotate(
        "", xy=(lon, lat), xytext=(lx, ly),
        arrowprops=dict(
            arrowstyle="-", color=col, lw=0.75,
            connectionstyle="arc3,rad=0.05",   # gentler curve than v1
        ),
        zorder=5,
    )
    # Label box
    ax.text(
        lx - 0.5, ly,
        f"{orig}   n = {count}",
        ha="right", va="center",
        fontsize=9.5, fontweight="bold", color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=col, linewidth=0.9, alpha=0.93),
        zorder=6,
    )

# 6) Legend
legend_items = [
    mpatches.Patch(color=PALETTE["blue"],   label="1–5 publications"),
    mpatches.Patch(color=PALETTE["teal"],   label="6–11 publications"),
    mpatches.Patch(color=PALETTE["orange"], label="≥ 12 publications"),
    mpatches.Patch(facecolor="#DDE6ED", edgecolor="#A8B8C2",
                   label="No publications"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=10,
          frameon=True, edgecolor="#CCCCCC", framealpha=0.93,
          title="Publications per country", title_fontsize=9)

# 7) Axes and grid
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 85)
ax.set_xticks(range(-150, 180, 30))
ax.set_yticks(range(-60, 90, 30))
ax.set_xticklabels([f"{x}°" for x in range(-150, 180, 30)],
                   color="#999", fontsize=7.5)
ax.set_yticklabels([f"{y}°" for y in range(-60, 90, 30)],
                   color="#999", fontsize=7.5)
ax.grid(True, linewidth=0.25, alpha=0.4, color="white", zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor("#CCCCCC")

# No title (removed per request)

plt.tight_layout()
save_fig(OUT)
