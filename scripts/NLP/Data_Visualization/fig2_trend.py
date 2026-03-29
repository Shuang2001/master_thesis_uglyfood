import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COL_BLUE   = "#5B8DB8"
COL_ORANGE = "#E8956D"
COL_TEAL   = "#5FBFCF"
DARK       = "#2C2C2C"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

df = pd.read_csv(r"C:\masterarbeit\uglyfood_nlp\thesis_figures\data_processed\bertopic_doc_topics.csv",
                 encoding="cp1252", on_bad_lines="skip")

trend  = df["year"].value_counts().sort_index()
years  = trend.index.to_numpy()
counts = trend.values

n_years = years[-1] - years[0]
cagr    = (counts[-1] / counts[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

fig, ax = plt.subplots(figsize=(11, 6))

ax.fill_between(years, counts, alpha=0.15, color=COL_BLUE)
ax.plot(years, counts, color=COL_BLUE, lw=2.5, zorder=3)
ax.scatter(years, counts, color=COL_BLUE, s=50, zorder=4)

peak_yr  = years[np.argmax(counts)]
peak_cnt = counts.max()

ax.scatter([peak_yr], [peak_cnt],
           color=COL_ORANGE, s=120, zorder=5, edgecolors="white", lw=1.5)
ax.annotate(f"Peak: {int(peak_yr)}\n(n = {peak_cnt})",
            xy=(peak_yr, peak_cnt),
            xytext=(peak_yr - 1.5, peak_cnt + 0.8),
            fontsize=9.5, color=COL_ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=COL_ORANGE, lw=1.3))

ax.axvspan(years[-3], years[-1] + 0.4,
           color=COL_TEAL, alpha=0.08, zorder=0)
ax.text(years[-3] + 0.1, 0.5,
        "Recent\ngrowth", fontsize=8.5, color=COL_TEAL,
        va="bottom", style="italic")


ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Number of Publications", fontsize=11)
# NO title
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax.set_ylim(0, peak_cnt * 1.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(r"C:\masterarbeit\uglyfood_nlp\thesis_figures\Figure2_fixed.png",
            dpi=200, bbox_inches="tight")
print("saved")
