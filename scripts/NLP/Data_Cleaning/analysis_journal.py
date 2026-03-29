import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# 读取数据
df = pd.read_csv("data_processed/final_dataset.csv")

# 统计Top15期刊
journal_counts = df["journal"].value_counts().head(15)
journal_counts = journal_counts.sort_values()

# 设置学术风格
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# 自动换行函数（避免期刊名被截断）
def wrap_labels(labels, width=35):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

wrapped_labels = wrap_labels(journal_counts.index, width=35)

# 绘图
plt.figure(figsize=(10,8))

bars = plt.barh(wrapped_labels, journal_counts.values)

plt.xlabel("Number of Publications")
plt.title("Top 15 Journals in Ugly/Imperfect Produce Consumer Research")

# 在柱子末端标数字
for i, v in enumerate(journal_counts.values):
    plt.text(v + 0.2, i, str(v), va='center')

plt.tight_layout()
plt.savefig("data_processed/top_journals_academic.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure: top_journals_academic.png")