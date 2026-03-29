import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ========= Config =========
DATA_PATH = "data_processed/bertopic_doc_topics.csv"
OUT_DIR = "data_processed"

# ========= Topic标签（你手动命名） =========
TOPIC_LABELS = {
    -1: "Outliers (unclassified)",
     0: "T0: Appearance Cues &\nQuality Inference",
     1: "T1: Pro-environmental Values\n& Behavioral Norms",
     2: "T2: Anthropomorphic Framing\n& Aesthetic Design",
     3: "T3: Sustainable Labels\n& Green Signaling",
     4: "T4: Retail & Price\nIntervention Mechanisms",
     5: "T5: Marketing Strategy\n& Asian Context",
     6: "T6: Supply Chain\n& Market Structure (peripheral)",
}

TOPIC_COLORS = {
    -1: "#cccccc",
     0: "#2980b9",
     1: "#27ae60",
     2: "#e74c3c",   # 红色 = 你的研究主题
     3: "#8e44ad",
     4: "#e67e22",
     5: "#16a085",
     6: "#95a5a6",
}

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH, encoding='latin1')
df_main = df[df['topic'] != -1].copy()

# =========================================================
# 图1: Topic × 文献数量柱状图 (Figure 6)
# =========================================================
def plot_topic_counts():
    topic_counts = df['topic'].value_counts().sort_index()
    # 排除outlier(-1)，单独标注
    topics = [t for t in topic_counts.index if t != -1]
    counts = [topic_counts[t] for t in topics]
    labels = [TOPIC_LABELS[t] for t in topics]
    colors = [TOPIC_COLORS[t] for t in topics]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(labels, counts, color=colors, edgecolor='white', height=0.65)

    # 数值标签
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f'n={val}', va='center', ha='left', fontsize=11, fontweight='bold')

    # 高亮T2（你的研究主题）
    ax.axhline(y=labels.index(TOPIC_LABELS[2]), color='#e74c3c', linewidth=0, alpha=0)
    t2_idx = labels.index(TOPIC_LABELS[2])
    ax.annotate('← This study\'s focus',
                xy=(counts[t2_idx] + 0.2, t2_idx),
                xytext=(counts[t2_idx] + 3, t2_idx),
                fontsize=9, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # 分类线：核心/中等/边缘
    ax.axvline(x=11, color='gray', linestyle='--', alpha=0.4, lw=1)
    ax.text(11.1, -0.6, 'Core threshold (n≥11)', fontsize=8, color='gray', alpha=0.7)

    ax.set_xlabel("Number of Publications", fontsize=12)
    ax.set_title("Figure 6: BERTopic Thematic Clusters in Ugly/Suboptimal Food Research\n"
                 "(n=84 core papers; T2 = focal topic of this study)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(counts) * 1.35)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Figure6_topic_counts.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Figure6_topic_counts.png")

# =========================================================
# 图2: Topic × 年份分布热力图 (Figure 7)
# =========================================================
def plot_topic_year_heatmap():
    topics = sorted([t for t in df['topic'].unique() if t != -1])
    years = sorted(df['year'].dropna().unique().astype(int))

    matrix = pd.DataFrame(0, index=topics, columns=years)
    for _, row in df.iterrows():
        t = row['topic']
        y = row['year']
        if t != -1 and not pd.isna(y):
            matrix.loc[t, int(y)] += 1

    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix.values, cmap=cmap, aspect='auto', vmin=0)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([TOPIC_LABELS[t].replace('\n', ' ') for t in topics], fontsize=9)

    # 数字标注
    for i, t in enumerate(topics):
        for j, y in enumerate(years):
            val = matrix.loc[t, y]
            if val > 0:
                color = 'white' if val >= 3 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    # 高亮T2行
    t2_idx = topics.index(2)
    ax.add_patch(plt.Rectangle((-0.5, t2_idx - 0.5), len(years), 1,
                                fill=False, edgecolor='#e74c3c', lw=2.5))

    plt.colorbar(im, ax=ax, label='Number of Publications', shrink=0.8)
    ax.set_title("Figure 7: Publication Distribution by Topic and Year\n"
                 "(Red border = T2 Anthropomorphic Framing — this study's focal topic)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Publication Year", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Figure7_topic_year_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Figure7_topic_year_heatmap.png")

# =========================================================
# 图3: 理论框架使用频次 (Figure 8)
# =========================================================
def plot_theory_usage():
    theory_counts = df['Primary_Theory'].value_counts()
    # 把"None explicit"单独处理
    labeled = theory_counts[theory_counts.index != 'None explicit (abstract-level)']
    n_none = theory_counts.get('None explicit (abstract-level)', 0)

    labels = list(labeled.index)
    values = list(labeled.values)
    colors = ['#e74c3c' if 'Anthropo' in l else '#2980b9' for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va='center', ha='left', fontsize=11, fontweight='bold')

    # 注释：有多少篇没有明确理论
    ax.text(0.98, 0.02,
            f'Note: {n_none} papers did not explicitly state\na primary theoretical framework.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#cccccc'))

    patch_gap = mpatches.Patch(color='#e74c3c', label='Anthropomorphism (this study\'s theory)')
    patch_other = mpatches.Patch(color='#2980b9', label='Other established theories')
    ax.legend(handles=[patch_other, patch_gap], loc='lower right', fontsize=9)

    ax.set_xlabel("Number of Publications", fontsize=12)
    ax.set_title("Figure 8: Theoretical Frameworks Explicitly Applied\nin Ugly/Suboptimal Food Research (n=84)",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.4)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Figure8_theory_usage.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Figure8_theory_usage.png")

# =========================================================
# 图4: 国家分布 + 中国稀缺证明 (Figure 9)
# =========================================================
def plot_country_distribution():
    # 拆分多国家的行
    country_list = []
    for _, row in df.iterrows():
        if pd.isna(row['Country']):
            continue
        countries = [c.strip() for c in str(row['Country']).split(';')]
        for c in countries:
            if c:
                country_list.append({'country': c, 'topic': row['topic']})

    cdf = pd.DataFrame(country_list)
    country_counts = cdf['country'].value_counts().head(15)

    # 颜色：中国用橙色高亮，欧美用蓝色
    def get_color(c):
        if c == 'China':
            return '#e67e22'
        elif c in ['United States', 'Germany', 'Netherlands', 'Denmark',
                   'France', 'United Kingdom', 'Switzerland', 'Belgium', 'Sweden']:
            return '#2980b9'
        else:
            return '#95a5a6'

    colors = [get_color(c) for c in country_counts.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(country_counts.index, country_counts.values,
                   color=colors, edgecolor='white', height=0.65)

    for bar, val in zip(bars, country_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', ha='left', fontsize=11, fontweight='bold')

    patch_eu = mpatches.Patch(color='#2980b9', label='Western countries (dominant)')
    patch_cn = mpatches.Patch(color='#e67e22', label='China (underrepresented → research gap)')
    patch_other = mpatches.Patch(color='#95a5a6', label='Other regions')
    ax.legend(handles=[patch_eu, patch_cn, patch_other], loc='lower right', fontsize=9)

    # 计算欧美 vs 中国比例
    western = ['United States', 'Germany', 'Netherlands', 'Denmark',
               'France', 'United Kingdom', 'Switzerland', 'Belgium', 'Sweden']
    n_western = sum(country_counts.get(c, 0) for c in western)
    n_china = country_counts.get('China', 0)
    ax.text(0.98, 0.97,
            f'Western countries: {n_western} studies\nChina: {n_china} studies\n'
            f'Ratio: {n_western/max(n_china,1):.1f}:1',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#e67e22'))

    ax.set_xlabel("Number of Publications", fontsize=12)
    ax.set_title("Figure 9: Geographic Distribution of Ugly Food Research\n"
                 "(Western dominance; Chinese context underrepresented)",
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Figure9_country_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Figure9_country_distribution.png")

# =========================================================
# 图5: 机制 × 边界条件 — 研究空白热力图 (Figure 10)
# =========================================================
def plot_mechanism_gap():
    # 统计Mechanism_Tested的频次
    mech_counts = df['Mechanism_Tested'].value_counts()
    # 过滤掉 "Not specified" 和 "Attitudes / intentions (direct)"（太宽泛）
    exclude = ['Not specified', 'Attitudes / intentions (direct)']
    mech_filtered = mech_counts[~mech_counts.index.isin(exclude)]

    labels = list(mech_filtered.index)
    values = list(mech_filtered.values)
    # 高亮同情心
    colors = ['#e74c3c' if 'Sympathy' in l or 'Empathic' in l else '#2980b9' for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va='center', ha='left', fontsize=11, fontweight='bold')

    # 注释同情心
    symp_idx = next((i for i, l in enumerate(labels) if 'Sympathy' in l or 'Empathic' in l), None)
    if symp_idx is not None:
        ax.annotate('Sympathy as mediator:\nonly 1 study → gap',
                    xy=(values[symp_idx] + 0.1, symp_idx),
                    xytext=(values[symp_idx] + 3, symp_idx),
                    fontsize=9, color='#e74c3c', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    patch_gap = mpatches.Patch(color='#e74c3c', label='This study\'s mediator (Sympathy)')
    patch_other = mpatches.Patch(color='#2980b9', label='Other mechanisms')
    ax.legend(handles=[patch_other, patch_gap], loc='lower right', fontsize=9)

    ax.set_xlabel("Number of Publications", fontsize=12)
    ax.set_title("Figure 10: Mediating Mechanisms Tested in Ugly Food Research\n"
                 "(Sympathy/Empathy as mediator = severely understudied)",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.5)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Figure10_mechanism_gap.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Figure10_mechanism_gap.png")

# =========================================================
# 文字发现输出（写论文用）
# =========================================================
def print_findings():
    print("\n" + "="*60)
    print("📝 文字发现（直接写进论文）")
    print("="*60)

    # 拟人化核心IV
    n_anthr_iv = (df['Anthropomorphism_as_Core_IV'] == 'Yes').sum()
    n_total = len(df[df['topic'] != -1])
    print(f"\n【拟人化作为核心IV】")
    print(f"  仅 {n_anthr_iv}/{n_total} 篇将拟人化作为核心自变量 ({n_anthr_iv/n_total*100:.1f}%)")
    print(f"  → 写法：'Although anthropomorphism appears in {n_anthr_iv} studies,")
    print(f"    only {n_anthr_iv} explicitly position it as the primary IV.'")

    # 同情心机制
    symp = df['Mechanism_Tested'].str.contains('Sympathy|Empathic', na=False).sum()
    print(f"\n【同情心中介机制】")
    print(f"  同情心作为中介机制：仅 {symp} 篇")
    print(f"  → 写法：'Sympathy as a mediating mechanism remains virtually")
    print(f"    unexplored (n={symp}), representing a critical theoretical gap.'")

    # 中国情境
    china_count = df['Country'].str.contains('China', na=False).sum()
    print(f"\n【中国情境稀缺】")
    print(f"  中国样本文献：{china_count} 篇 / {len(df)} 篇 ({china_count/len(df)*100:.1f}%)")
    print(f"  → 写法：'Chinese-context studies account for only {china_count/len(df)*100:.0f}%")
    print(f"    of the corpus, indicating a geographic gap.'")

    # 核心主题分类
    print(f"\n【主题分类建议】")
    topic_counts = df[df['topic'] != -1]['topic'].value_counts().sort_index()
    for t, n in topic_counts.items():
        tier = "核心主题 ✅" if n >= 11 else ("中等主题 ➡" if n >= 7 else "边缘主题 ⚠")
        print(f"  {TOPIC_LABELS[t].replace(chr(10),' ')} → n={n} → {tier}")

    print("\n" + "="*60)

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("生成模块A可视化图表...\n")
    plot_topic_counts()
    plot_topic_year_heatmap()
    plot_theory_usage()
    plot_country_distribution()
    plot_mechanism_gap()
    print_findings()

    print("\n✅ 全部完成！生成了5张图：")
    print("  Figure6_topic_counts.png       — 主题文献分布")
    print("  Figure7_topic_year_heatmap.png — 主题×年份热力图")
    print("  Figure8_theory_usage.png       — 理论框架频次")
    print("  Figure9_country_distribution.png — 国家分布")
    print("  Figure10_mechanism_gap.png     — 机制研究空白")