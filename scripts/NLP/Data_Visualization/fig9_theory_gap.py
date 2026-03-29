import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

COL_ORANGE = "#E8956D"
COL_BLUE   = "#5B8DB8"
DARK       = "#2C2C2C"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

THEORIES = {
    "Theory of Planned\nBehavior (TPB)":    ["theory of planned behavior", "planned behaviour", "TPB"],
    "Signaling Theory":                      ["signal theory", "signaling theory", "signalling theory", "cue utilization"],
    "Prospect Theory":                       ["prospect theory", "loss aversion", "framing effect"],
    "Norm Activation\nModel (NAM)":          ["norm activation", "moral norm", "NAM model"],
    "Value-Belief-Norm (VBN)":               ["value belief norm", "VBN theory"],
    "Social Norm Theory":                    ["social norm", "descriptive norm", "injunctive norm"],
    "Attribution Theory":                    ["attribution theory", "causal attribution"],
    "Elaboration\nLikelihood (ELM)":         ["elaboration likelihood", "ELM model", "central route", "peripheral route"],
    "Anthropomorphism\nTheory ★":            ["anthropomorph", "anthropomorphism", "humaniz",
                                              "human-like feature", "face on food",
                                              "injured apple", "wounded", "personif"],
}

df = pd.read_csv("/mnt/user-data/uploads/bertopic_doc_topics.csv",
                 encoding="cp1252", on_bad_lines="skip", low_memory=False)

search_cols = [c for c in ["abstract", "title", "author_keywords", "index_keywords"]
               if c in df.columns]

def hit(df, cols, keywords):
    import pandas as pd
    mask = pd.Series([False] * len(df))
    for col in cols:
        txt = df[col].fillna("").astype(str).str.lower()
        for kw in keywords:
            mask |= txt.str.contains(kw.lower(), regex=False)
    return int(mask.sum())

counts = {t: hit(df, search_cols, kws) for t, kws in THEORIES.items()}
items  = sorted(counts.items(), key=lambda x: x[1], reverse=True)
labels = [x[0] for x in items]
values = [x[1] for x in items]

fig, ax = plt.subplots(figsize=(12, 7))

colors = [COL_ORANGE if ("Anthropomorph" in l or "★" in l) else COL_BLUE
          for l in labels]

bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="none", zorder=3)

for bar, val, col in zip(bars, values, colors):
    fw = "bold" if col == COL_ORANGE else "normal"
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left",
            fontsize=11, fontweight=fw, color=col)

anthr_idx = next((i for i, l in enumerate(labels)
                  if "Anthropomorph" in l or "★" in l), None)
if anthr_idx is not None and values[anthr_idx] == 0:
    ax.annotate("Research Gap\n(0 studies found)",
                xy=(0.3, anthr_idx),
                xytext=(max(values) * 0.45, anthr_idx),
                fontsize=10, color=COL_ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=COL_ORANGE, lw=1.5))



leg = [mpatches.Patch(color=COL_BLUE,   label="Established theories in ugly food research"),
       mpatches.Patch(color=COL_ORANGE,  label="Research gap — this study")]
ax.legend(handles=leg, loc="lower right", fontsize=10,
          frameon=True, edgecolor="#CCCCCC", framealpha=0.92)

ax.set_xlabel("Number of Publications", fontsize=11)
ax.set_xlim(0, max(values) * 1.30 + 1)
ax.invert_yaxis()

# vertical grid only
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=10)
ax.tick_params(axis="x", labelsize=10)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure9_fixed.png", dpi=200, bbox_inches="tight")
print("fig9 saved")
