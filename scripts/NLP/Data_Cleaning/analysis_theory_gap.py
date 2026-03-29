# analysis_theory_gap.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# ========= Config =========
DATA_PATH = "data_processed/final_dataset.csv"
OUT_DIR = "data_processed"
OUT_PNG = os.path.join(OUT_DIR, "Figure6_theory_gap.png")

# ========= 要搜索的理论关键词 =========
# 格式：{ "显示名称": ["搜索词1", "搜索词2", ...] }
THEORIES = {
    "Theory of Planned\nBehavior (TPB)": ["theory of planned behavior", "planned behaviour", "TPB"],
    "Signaling Theory": ["signal theory", "signaling theory", "signalling theory", "cue utilization"],
    "Prospect Theory": ["prospect theory", "loss aversion", "framing effect"],
    "Norm Activation\nModel (NAM)": ["norm activation", "moral norm", "NAM model"],
    "Value-Belief-Norm\n(VBN)": ["value belief norm", "VBN theory"],
    "Technology\nAcceptance (TAM)": ["technology acceptance", "TAM model"],
    "Social Norm Theory": ["social norm", "descriptive norm", "injunctive norm"],
    "Attribution Theory": ["attribution theory", "causal attribution"],
    "Elaboration\nLikelihood (ELM)": ["elaboration likelihood", "ELM model", "central route", "peripheral route"],
    "Anthropomorphism\n(拟人化) ★": ["anthropomorph", "anthropomorphism", "humaniz", "human-like feature",
                                      "face on food", "injured apple", "wounded", "personif"],
}

def search_theory(df, text_cols, keywords):
    """在指定列中搜索关键词，返回命中文献数"""
    mask = pd.Series([False] * len(df))
    for col in text_cols:
        if col not in df.columns:
            continue
        col_text = df[col].fillna('').astype(str).str.lower()
        for kw in keywords:
            mask = mask | col_text.str.contains(kw.lower(), regex=False)
    return int(mask.sum())

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} records")

    # 搜索范围：摘要 + 标题 + 关键词列
    search_cols = []
    for candidate in ["abstract", "Abstract", "AB", "title", "Title", "TI",
                       "keywords", "Keywords", "DE", "ID", "author_keywords"]:
        if candidate in df.columns:
            search_cols.append(candidate)
    print("Searching in columns:", search_cols)

    # 统计每个理论出现次数
    counts = {}
    for theory_name, keywords in THEORIES.items():
        n = search_theory(df, search_cols, keywords)
        counts[theory_name] = n
        print(f"  {theory_name.replace(chr(10), ' ')}: {n}")

    # 排序
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    # ========= 画图 =========
    fig, ax = plt.subplots(figsize=(12, 7))

    # 拟人化那条用红色高亮，其他用蓝色
    colors = []
    for label in labels:
        if "Anthropomorph" in label or "拟人化" in label:
            colors.append('#e74c3c')   # 红色 = 空白
        else:
            colors.append('#2980b9')   # 蓝色 = 已有研究

    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)

    # 数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=11, fontweight='bold')

    # 拟人化那行加注释箭头
    anthr_idx = next((i for i, l in enumerate(labels) if "Anthropomorph" in l or "拟人化" in l), None)
    if anthr_idx is not None and values[anthr_idx] == 0:
        ax.annotate('Research Gap\n(0 studies)',
                    xy=(0.2, anthr_idx),
                    xytext=(max(values) * 0.4, anthr_idx),
                    fontsize=10, color='#e74c3c', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # 图例
    patch_exist = mpatches.Patch(color='#2980b9', label='Established theories in ugly food research')
    patch_gap = mpatches.Patch(color='#e74c3c', label='Research gap (this study)')
    ax.legend(handles=[patch_exist, patch_gap], loc='lower right', fontsize=10)

    ax.set_xlabel("Number of Publications", fontsize=12)
    ax.set_title("Theoretical Frameworks in Ugly Food / Suboptimal Food Research\n"
                 "(Anthropomorphism Theory Remains Unexplored)", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.25 + 1)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {OUT_PNG}")

    # 顺便打印文字发现供你写论文用
    print("\n=== 文字发现（可直接写进论文）===")
    top3 = sorted_items[:3]
    print(f"最常用理论: {top3[0][0].replace(chr(10),' ')} (n={top3[0][1]}), "
          f"{top3[1][0].replace(chr(10),' ')} (n={top3[1][1]}), "
          f"{top3[2][0].replace(chr(10),' ')} (n={top3[2][1]})")
    anthr_count = counts.get(list(THEORIES.keys())[-1], 0)
    print(f"拟人化理论出现次数: {anthr_count} → 证明研究空白")

if __name__ == "__main__":
    main()