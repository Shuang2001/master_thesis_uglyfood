import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# ====== 0) Load ======
df = pd.read_csv("data_processed/final_dataset.csv")

# ====== 1) Parse Author Keywords ======
keywords = df["author_keywords"].dropna().astype(str)

all_keywords = []
for row in keywords:
    parts = re.split(r";|,", row)
    for p in parts:
        k = p.strip().lower()
        if k:
            all_keywords.append(k)

# ====== 2) Normalize / Merge synonyms (extend as needed) ======
replace_map = {
    "purchase intention": "purchase intention",
    "buying intention": "purchase intention",
    "willingness to pay": "wtp",
    "willingness-to-pay": "wtp",
    "consumer behavior": "consumer behaviour",
    "consumer behaviour": "consumer behaviour",
    "ugly food": "ugly food",
    "imperfect food": "imperfect food",
    "imperfect produce": "imperfect produce",
}

normalized = [replace_map.get(k, k) for k in all_keywords]

# Optional: remove very generic noise terms if they appear
stop_terms = {"study", "food", "consumer", "consumers"}  # 先保守；如果你不想删，就清空这个集合
normalized = [k for k in normalized if k not in stop_terms]

# ====== 3) Count & select Top N ======
counter = Counter(normalized)
top_n = 20
top = counter.most_common(top_n)

labels = [x[0] for x in top][::-1]
values = [x[1] for x in top][::-1]

# ====== 4) Plot: match Figure2 style (clean, academic) ======
plt.figure(figsize=(12, 8))

# single color (no gradient), consistent with Figure2 default blue
bars = plt.barh(labels, values)

plt.xlabel("Frequency")
plt.title("Top 20 Author Keywords in the Final Corpus")

# light grid like your Figure2
plt.grid(True, axis="x", linewidth=0.6, alpha=0.35)

# clean spines (optional but makes it more publication-like)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# value labels at bar end (same as your Figure2 journal plot style)
for i, v in enumerate(values):
    plt.text(v + 0.1, i, str(v), va="center", fontsize=10)

plt.tight_layout()
plt.savefig("data_processed/top_keywords.png", dpi=300)
plt.show()

print("Top keywords:", top)