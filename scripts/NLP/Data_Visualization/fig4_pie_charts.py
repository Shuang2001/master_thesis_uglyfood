import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

C = [
    "#5B8DB8",  # 蓝
    "#6AAF6A",  # 绿
    "#E8956D",  # 橙
    "#9B85B5",  # 紫
    "#C98BAD",  # 粉
    "#5FBFCF",  # 青
    "#8C8C8C",  # 备用灰
    "#BCBD22",  # 备用
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA = "/mnt/user-data/uploads/bertopic_doc_topics.csv"
df = pd.read_csv(DATA, encoding="cp1252", on_bad_lines="skip")

METHOD_PATTERNS = {
    "Experimental Study":         r"\bexperim\w*\b|\bmanipulat\w*\b|\bbetween.subject\b",
    "Online Survey":              r"\bonline survey\b|\bquestionnaire\b|\bself.report\b",
    "Choice Experiment /\nWTP":   r"\bchoice experiment\b|\bwillingness.to.pay\b|\bwtp\b|\bdiscrete choice\b",
    "Lab/Field\nExperiment":      r"\blab\w* experiment\b|\bfield experiment\b|\bin.store\b",
    "Meta-analysis /\nReview":    r"\bmeta.analy\w*\b|\bsystematic review\b|\bliterature review\b",
    "Qualitative /\nInterview":   r"\binterview\b|\bfocus group\b|\bqualitative\b|\bethnograph\w*\b",
    "Other / Mixed":              r".",
}

def detect_method(text):
    t = str(text).lower()
    for method, pat in METHOD_PATTERNS.items():
        if method == "Other / Mixed":
            continue
        if re.search(pat, t):
            return method
    return "Other / Mixed"

df["method"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).apply(detect_method)
method_counts = df["method"].value_counts()

TOPIC_LABELS = {
    -1: "Uncategorised",
     0: "Sensory & Quality\nPerception",
     1: "Environmental Values\n& Behaviour",
     2: "Anthropomorphism &\nAesthetic Imperfection",
     3: "Sustainability &\nOrganic Labels",
     4: "Packaging, Price\n& Retailer",
     5: "Marketing Strategy\n& Positioning",
     6: "Farmer/Market\n& Standards",
}
df["topic_label"] = df["topic"].map(TOPIC_LABELS).fillna("Uncategorised")
topic_counts = df["topic_label"].value_counts()
topic_counts = topic_counts[topic_counts.index != "Uncategorised"]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

OUTSIDE_THRESHOLD = 0.06

def draw_pie(ax, counts, title, colors):
    labels = counts.index.tolist()
    sizes  = counts.values
    total  = sizes.sum()
    shares = sizes / total

    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f"{p:.1f}%\n(n={int(round(p*total/100))})",
        colors=colors[:len(sizes)],
        startangle=140,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )

    for i, (at, share) in enumerate(zip(autotexts, shares)):
        if share < OUTSIDE_THRESHOLD:
            at.set_visible(False)
            theta = np.deg2rad((wedges[i].theta1 + wedges[i].theta2) / 2)
            r_inner = 0.72
            r_outer = 1.18
            x_out = r_outer * np.cos(theta)
            y_out = r_outer * np.sin(theta)
            pct_text = f"{share*100:.1f}%\n(n={sizes[i]})"
            ax.annotate(
                pct_text,
                xy=(r_inner * np.cos(theta), r_inner * np.sin(theta)),
                xytext=(x_out, y_out),
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
                ha="center", va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#555555",
                    lw=0.9,
                    connectionstyle="arc3,rad=0.0",
                ),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )
        else:
            at.set_fontsize(8.5)
            at.set_color("white")
            at.set_fontweight("bold")

    legend_labels = [f"{l}  (n={s})" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels,
              loc="lower center",
              bbox_to_anchor=(0.5, -0.30),
              ncol=2,
              fontsize=10,        # 图例字号放大
              frameon=False)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)  # 标题放大

draw_pie(axes[0], method_counts,
         "Research Methods Distribution\n(n = 84 studies)", C)

draw_pie(axes[1], topic_counts,
         "Research Topics Distribution\n(BERTopic, n = 77 classified studies)", C)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Figure4_pie_charts_final.png",
            dpi=200, bbox_inches="tight")
print("saved")
