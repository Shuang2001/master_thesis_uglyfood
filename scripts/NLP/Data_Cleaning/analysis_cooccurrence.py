import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import re
import itertools
import math

# ========= Settings (consistent academic style) =========
TOP_N = 18
MIN_EDGE = 2          # core网络更稀疏时用1更稳；想更“团簇”可改2
SEED = 42
FIGSIZE = (14, 9)
DPI = 300

IN_PATH = "data_processed/final_dataset.csv"

OUT_FIG_CORE = "data_processed/Figure4_keyword_cooccurrence_network.png"
OUT_FIG_STAR = "data_processed/Figure4_star_keyword_cooccurrence_network.png"

OUT_MAT_CORE = "data_processed/keyword_cooccurrence_top20_core.csv"
OUT_MAT_STAR = "data_processed/keyword_cooccurrence_top20_star.csv"

# ========= Helpers =========
def parse_keywords(s: str):
    parts = re.split(r";|,", str(s))
    out = []
    for p in parts:
        k = p.strip().lower()
        if k:
            out.append(k)
    return out

replace_map = {
    "buying intention": "purchase intention",
    "purchase intentions": "purchase intention",
    "purchase intention": "purchase intention",
    "willingness to pay": "wtp",
    "willingness-to-pay": "wtp",
    "consumer behavior": "consumer behaviour",
    "consumer behaviour": "consumer behaviour",
    "suboptimal foods": "suboptimal food",
    "suboptimal products": "suboptimal food products",
}

def build_cooccurrence(df, remove_umbrella_terms: bool):
    # 基础噪声词
    stop_terms = {"study"}

    # 关键：去掉“议题框架/伞状词”，让机制结构显现
    if remove_umbrella_terms:
        stop_terms.update({"food waste", "consumer food waste"})

    papers = []
    for s in df["author_keywords"].dropna().astype(str):
        ks = parse_keywords(s)
        ks = [replace_map.get(k, k) for k in ks]
        ks = [k for k in ks if k not in stop_terms]
        ks = list(dict.fromkeys(ks))  # unique within paper
        if len(ks) >= 2:
            papers.append(ks)

    # 词频 -> Top N
    all_kw = list(itertools.chain.from_iterable(papers))
    freq = Counter(all_kw)
    top_keywords = [k for k, _ in freq.most_common(TOP_N)]
    top_set = set(top_keywords)

    # 共现计数
    co = Counter()
    for ks in papers:
        ks2 = [k for k in ks if k in top_set]
        ks2 = sorted(set(ks2))
        for a, b in itertools.combinations(ks2, 2):
            co[(a, b)] += 1

    # 共现矩阵导出
    mat = pd.DataFrame(0, index=top_keywords, columns=top_keywords, dtype=int)
    for (a, b), v in co.items():
        mat.loc[a, b] = v
        mat.loc[b, a] = v

    # 建图
    G = nx.Graph()
    for k in top_keywords:
        G.add_node(k, freq=freq[k])

    for (a, b), v in co.items():
        if v >= MIN_EDGE:
            G.add_edge(a, b, weight=v)

    # 移除孤立点（避免漂浮点和无意义标签）
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G, mat, len(isolates)

def draw_graph(G, title, out_path):
    if G.number_of_nodes() == 0:
        print("Graph is empty, skip:", out_path)
        return

    n = max(G.number_of_nodes(), 1)
    k = 1.2 / math.sqrt(n)
    pos = nx.spring_layout(G, seed=SEED, k=k, iterations=300)

    plt.figure(figsize=FIGSIZE)

    node_sizes = [120 + 35 * G.nodes[n]["freq"] for n in G.nodes()]
    edge_widths = [0.6 + 0.9 * G.edges[e]["weight"] for e in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.30)

    # 标签策略：Top 15 频次 + 度>=4
    freq_sorted = sorted(G.nodes(), key=lambda x: G.nodes[x]["freq"], reverse=True)
    label_nodes = set(freq_sorted[:15])
    for node in G.nodes():
        if G.degree(node) >= 4:
            label_nodes.add(node)

    for node in label_nodes:
        x, y = pos[node]
        plt.text(
            x, y, node,
            fontsize=10,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
        )

    plt.title(title)
    plt.grid(True, linewidth=0.6, alpha=0.25)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.show()
    print("Saved figure:", out_path)

# ========= Run =========
df = pd.read_csv(IN_PATH, low_memory=False)

# 1) Star network (keep umbrella terms)
G_star, mat_star, iso_star = build_cooccurrence(df, remove_umbrella_terms=False)
mat_star.to_csv(OUT_MAT_STAR, encoding="utf-8-sig")
draw_graph(G_star, f"Keyword Co-occurrence Network (Top {TOP_N} Author Keywords) - Star", OUT_FIG_STAR)
print("Saved matrix:", OUT_MAT_STAR, "| isolates removed =", iso_star)

# 2) Core network (remove umbrella terms) -> main Figure4
G_core, mat_core, iso_core = build_cooccurrence(df, remove_umbrella_terms=True)
mat_core.to_csv(OUT_MAT_CORE, encoding="utf-8-sig")
draw_graph(G_core, f"Keyword Co-occurrence Network (Top {TOP_N} Author Keywords) - Core (umbrella terms removed)", OUT_FIG_CORE)
print("Saved matrix:", OUT_MAT_CORE, "| isolates removed =", iso_core)