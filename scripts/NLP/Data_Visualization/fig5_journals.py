import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

COL_ORANGE = "#E8956D"
COL_BLUE   = "#5B8DB8"
DARK       = "#2C2C2C"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

df = pd.read_csv("/mnt/user-data/uploads/bertopic_doc_topics.csv",
                 encoding="cp1252", on_bad_lines="skip")

journal_counts = df["journal"].value_counts().head(15).sort_values()
labels  = ['\n'.join(textwrap.wrap(j, 38)) for j in journal_counts.index]
values  = journal_counts.values
max_val = values.max()

top3_thr = sorted(values)[-3]
colors   = [COL_ORANGE if v >= top3_thr else COL_BLUE for v in values]

fig, ax = plt.subplots(figsize=(11, 8))

bars = ax.barh(labels, values, color=colors, height=0.65, edgecolor="none", zorder=3)

for bar, v, col in zip(bars, values, colors):
    fw = "bold" if col == COL_ORANGE else "normal"
    ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=10, color=col, fontweight=fw)

# legend
leg = [mpatches.Patch(color=COL_ORANGE, label="Top 3 journals"),
       mpatches.Patch(color=COL_BLUE,   label="Other journals")]
ax.legend(handles=leg, loc="lower right", fontsize=10,
          frameon=True, edgecolor="#CCCCCC", framealpha=0.92)

ax.set_xlabel("Number of Publications", fontsize=11)
ax.set_xlim(0, max_val * 1.22)

# vertical grid only, no horizontal
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=10)
ax.tick_params(axis="x", labelsize=10)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure5_fixed.png", dpi=200, bbox_inches="tight")
print("fig5 saved")
