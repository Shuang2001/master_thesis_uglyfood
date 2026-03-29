import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import re

COL_BLUE   = "#5B8DB8"
COL_ORANGE = "#E8956D"
COL_TEAL   = "#5FBFCF"
DARK       = "#2C2C2C"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

df = pd.read_csv("/mnt/user-data/uploads/bertopic_doc_topics.csv",
                 encoding="cp1252", on_bad_lines="skip")

replace_map = {
    "buying intention":    "purchase intention",
    "purchase intentions": "purchase intention",
    "willingness to pay":  "wtp",
    "willingness-to-pay":  "wtp",
    "consumer behavior":   "consumer behaviour",
}
stop_terms = {"study", "food", "consumer", "consumers"}

all_kw = []
for row in df["author_keywords"].dropna().astype(str):
    for p in re.split(r";|,", row):
        k = p.strip().lower()
        k = replace_map.get(k, k)
        if k and k not in stop_terms:
            all_kw.append(k)

counter = Counter(all_kw)
top     = counter.most_common(20)
labels  = [x[0] for x in top][::-1]
values  = [x[1] for x in top][::-1]
max_v   = max(values)

HIGHLIGHT_TERMS = {"anthropomorphism", "imperfect produce", "ugly food",
                   "ugly produce", "imperfect food"}
top5_threshold = sorted(values)[-5]

colors = []
for lbl, v in zip(labels, values):
    if lbl in HIGHLIGHT_TERMS:
        colors.append(COL_TEAL)
    elif v >= top5_threshold:
        colors.append(COL_ORANGE)
    else:
        colors.append(COL_BLUE)

fig, ax = plt.subplots(figsize=(12, 9))

bars = ax.barh(labels, values, color=colors, height=0.65, edgecolor="none")

for bar, v, col in zip(bars, values, colors):
    fw = "bold" if col != COL_BLUE else "normal"
    ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=10, color=col, fontweight=fw)

# ── Proper legend inside axes ──────────────────────────────────────────────
leg_handles = [
    mpatches.Patch(color=COL_ORANGE, label="High-frequency terms (top 5)"),
    mpatches.Patch(color=COL_TEAL,   label="Topic-relevant terms"),
    mpatches.Patch(color=COL_BLUE,   label="Other keywords"),
]
ax.legend(handles=leg_handles,
          loc="lower right",
          fontsize=10,
          frameon=True,
          edgecolor="#CCCCCC",
          framealpha=0.92)

ax.set_xlabel("Frequency", fontsize=11)
# NO title
ax.set_xlim(0, max_v * 1.22)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure6_fixed.png",
            dpi=200, bbox_inches="tight")
print("saved")
