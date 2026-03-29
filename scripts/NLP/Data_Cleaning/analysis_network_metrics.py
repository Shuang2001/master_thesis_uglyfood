import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter

# ========= Settings =========
TOP_N = 18
MIN_EDGE = 2
DATA_PATH = "data_processed/final_dataset.csv"

# ========= Load Data =========
df = pd.read_csv(DATA_PATH, low_memory=False)

# 如果你的列名不同，在这里改
keywords_col = "author_keywords"

# ========= Preprocess =========
def clean_keywords(x):
    if pd.isna(x):
        return []
    return [k.strip().lower() for k in str(x).split(";")]

df["kw_list"] = df[keywords_col].apply(clean_keywords)

# ========= Top N Keywords =========
all_keywords = [kw for sublist in df["kw_list"] for kw in sublist]
kw_freq = Counter(all_keywords)

top_keywords = [k for k, _ in kw_freq.most_common(TOP_N)]

# 🔴 关键步骤：删除 umbrella term
if "food waste" in top_keywords:
    top_keywords.remove("food waste")

# ========= Build Graph =========
G = nx.Graph()

for kws in df["kw_list"]:
    kws = [k for k in kws if k in top_keywords]
    for k1, k2 in combinations(kws, 2):
        if G.has_edge(k1, k2):
            G[k1][k2]["weight"] += 1
        else:
            G.add_edge(k1, k2, weight=1)

# 删除弱连接
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < MIN_EDGE]
G.remove_edges_from(edges_to_remove)

# 删除孤立节点
G.remove_nodes_from(list(nx.isolates(G)))

# ========= Metrics =========
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
density = nx.density(G)

results = pd.DataFrame({
    "keyword": list(G.nodes()),
    "degree_centrality": [degree_centrality[n] for n in G.nodes()],
    "betweenness_centrality": [betweenness_centrality[n] for n in G.nodes()]
})

results = results.sort_values("degree_centrality", ascending=False)

print("\n=== Network Density ===")
print(density)

print("\n=== Centrality Ranking ===")
print(results)