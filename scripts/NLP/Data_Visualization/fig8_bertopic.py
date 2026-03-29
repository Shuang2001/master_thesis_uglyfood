import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ast, math

# ── Palette ────────────────────────────────────────────────────────────────
C = ["#5B8DB8","#6AAF6A","#E8956D","#9B85B5","#C98BAD","#5FBFCF","#8C8C8C","#BCBD22"]
DARK = "#2C2C2C"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

INFO_PATH = "/mnt/user-data/uploads/bertopic_topic_info_final.csv"

TOPIC_LABELS = {
     0: "Sensory & Quality Perception",
     1: "Environmental Values & Behaviour",
     2: "Anthropomorphism & Aesthetic Imperfection",
     3: "Sustainability & Organic Labels",
     4: "Packaging, Price & Retailer",
     5: "Marketing Strategy & Positioning",
     6: "Farmer/Market & Standards",
}

df = pd.read_csv(INFO_PATH, encoding="cp1252")
df.columns = [c.replace("ï»¿", "").strip() for c in df.columns]
df = df[df["Topic"] != -1].reset_index(drop=True)

def parse_rep(s):
    try:
        return ast.literal_eval(s)[:8]
    except Exception:
        return [x.strip().strip("'\"") for x in str(s).strip("[]").split(",")][:8]

df["kw_list"] = df["Representation"].apply(parse_rep)

n_topics = len(df)
ncols    = 2
nrows    = math.ceil(n_topics / ncols)

fig = plt.figure(figsize=(14, nrows * 2.8 + 1.5))
gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

for idx, row in df.iterrows():
    tid       = int(row["Topic"])
    label     = TOPIC_LABELS.get(tid, f"Topic {tid}")
    count     = int(row["Count"])
    kws       = row["kw_list"]
    bar_color = C[idx % len(C)]   # uniform colour treatment, no special case

    r, c = divmod(idx, ncols)
    ax   = fig.add_subplot(gs[r, c])

    y = range(len(kws))
    ax.barh(list(y), [1] * len(kws),
            color=bar_color, alpha=0.85, edgecolor="none", height=0.65)

    for yi, kw in enumerate(kws):
        ax.text(0.03, yi, kw,
                va="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xlim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # all titles in dark, no highlight, no background tint, no annotation
    ax.set_title(f"Topic {tid}: {label}\n(n = {count})",
                 fontsize=9.5, fontweight="bold",
                 color=DARK, pad=6)

# NO suptitle
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure8_fixed.png",
            dpi=200, bbox_inches="tight")
print("saved")
