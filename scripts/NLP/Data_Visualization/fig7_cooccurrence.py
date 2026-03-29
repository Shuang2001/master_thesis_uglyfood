import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import networkx as nx
from collections import Counter
import itertools, math, re
import numpy as np

PAL = {
    "orange": "#E8956D",
    "teal":   "#5FBFCF",
    "blue":   "#5B8DB8",
    "purple": "#9B85B5",
    "grey":   "#AAAAAA",
    "dark":   "#2C2C2C",
    "bg":     "#F8FAFC",
}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

DATA  = "/mnt/user-data/uploads/bertopic_doc_topics.csv"
TOP_N = 18
SEED  = 42

df = pd.read_csv(DATA, encoding="cp1252", on_bad_lines="skip", low_memory=False)

REPLACE = {
    "buying intention":    "purchase intention",
    "purchase intentions": "purchase intention",
    "willingness to pay":  "wtp",
    "willingness-to-pay":  "wtp",
    "consumer behavior":   "consumer behaviour",
    "consumer behaviour":  "consumer behaviour",
    "suboptimal foods":    "suboptimal food",
}
STOP_BASE = {"study"}

def parse_papers(df, extra_stop=set()):
    stop = STOP_BASE | extra_stop
    papers = []
    for s in df["author_keywords"].dropna().astype(str):
        ks = [p.strip().lower() for p in re.split(r";|,", s) if p.strip()]
        ks = [REPLACE.get(k, k) for k in ks]
        ks = [k for k in ks if k not in stop]
        ks = list(dict.fromkeys(ks))
        if len(ks) >= 2:
            papers.append(ks)
    return papers

def build_graph(papers, freq, top_n, min_edge):
    top_kw  = [k for k, _ in freq.most_common(top_n)]
    top_set = set(top_kw)
    co = Counter()
    for ks in papers:
        ks2 = sorted(set(k for k in ks if k in top_set))
        for a, b in itertools.combinations(ks2, 2):
            co[(a, b)] += 1
    G = nx.Graph()
    for k in top_kw:
        G.add_node(k, freq=freq[k])
    for (a, b), v in co.items():
        if v >= min_edge:
            G.add_edge(a, b, weight=v)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def tier(node, sorted_nodes):
    i = sorted_nodes.index(node)
    if i == 0:  return "top1"
    if i < 4:   return "top"
    if i < 8:   return "mid"
    return "base"

def ncolor(t):
    return {"top1": PAL["orange"], "top": PAL["teal"],
            "mid":  PAL["blue"],   "base": PAL["purple"]}[t]

def nsize(node, sorted_nodes, deg_cen, freq_map):
    t    = tier(node, sorted_nodes)
    base = 300 + 80 * freq_map[node] + 2500 * deg_cen[node]
    return base * {"top1": 2.2, "top": 1.5, "mid": 1.0, "base": 0.7}[t]

# ── smart label placement ──────────────────────────────────────────────────
# Estimate node radius in data coords given scatter markersize (pt²)
def node_radius_data(s_pt2, ax, fig):
    """Convert scatter s (pt²) to approximate radius in data coords."""
    # s = area in pt², radius in pt = sqrt(s/pi)
    r_pt = math.sqrt(s_pt2 / math.pi)
    # Convert pt → display (pixels) → data
    disp_per_pt = fig.dpi / 72.0
    r_disp = r_pt * disp_per_pt
    ax_disp_w = ax.get_window_extent().width
    ax_data_w = ax.get_xlim()[1] - ax.get_xlim()[0]
    return r_disp / ax_disp_w * ax_data_w

CANDIDATES = [
    (0,  1), (0, -1), (1,  0), (-1,  0),
    (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7),
    (0.4, 1), (-0.4, 1), (0.4, -1), (-0.4, -1),
    (1, 0.4), (-1, 0.4), (1, -0.4), (-1, -0.4),
]

def approx_text_size(text, fontsize, ax, fig):
    """Rough text bbox in data coords (width, height)."""
    chars   = max(len(line) for line in text.split("\n"))
    n_lines = text.count("\n") + 1
    pt_per_char_w = fontsize * 0.55
    pt_per_char_h = fontsize * 1.25
    w_pt = chars * pt_per_char_w
    h_pt = n_lines * pt_per_char_h
    disp_per_pt = fig.dpi / 72.0
    ax_win  = ax.get_window_extent()
    ax_data_w = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_data_h = ax.get_ylim()[1] - ax.get_ylim()[0]
    w_data = w_pt * disp_per_pt / ax_win.width  * ax_data_w
    h_data = h_pt * disp_per_pt / ax_win.height * ax_data_h
    return w_data, h_data

def place_labels(nodes, pos, sizes, sorted_nodes, deg_cen, freq_map, ax, fig,
                 reserved_bboxes=None):
    """
    For each node pick the candidate direction that:
      1. keeps the label close (distance = node_radius + small padding)
      2. avoids overlapping other nodes' centres
      3. avoids overlapping previously placed label bboxes
      4. avoids overlapping reserved_bboxes (e.g. centrality text box)
    Returns dict node -> (tx, ty, label_bbox_in_data)
    """
    placements = {}   # node -> (tx, ty)
    # pre-seed with any fixed UI regions (centrality box, legend area)
    placed_bboxes = list(reserved_bboxes) if reserved_bboxes else []

    # process high-centrality nodes first (they get priority)
    order = sorted(nodes, key=lambda n: deg_cen[n], reverse=True)

    for node in order:
        x, y   = pos[node]
        s_pt2  = sizes[nodes.index(node)]
        t      = tier(node, sorted_nodes)
        fsize  = 13 if t in ("top1", "top") else 11

        r_data = node_radius_data(s_pt2, ax, fig)
        pad    = r_data * 0.25          # small gap between node edge and label
        dist   = r_data + pad

        tw, th = approx_text_size(node, fsize, ax, fig)

        best_pos   = None
        best_score = 1e18

        for dx, dy in CANDIDATES:
            # normalise direction
            norm = math.hypot(dx, dy)
            udx, udy = dx / norm, dy / norm

            # label anchor (centre of text box)
            tx = x + udx * (dist + tw / 2)
            ty = y + udy * (dist + th / 2)

            # text bbox
            bx0, by0 = tx - tw/2, ty - th/2
            bx1, by1 = tx + tw/2, ty + th/2

            # penalty 1: overlap with other node centres
            node_penalty = 0
            for other in nodes:
                if other == node:
                    continue
                ox, oy = pos[other]
                si     = sizes[nodes.index(other)]
                or_    = node_radius_data(si, ax, fig)
                # does bbox overlap the node circle?
                cx = max(bx0, min(ox, bx1))
                cy = max(by0, min(oy, by1))
                d  = math.hypot(cx - ox, cy - oy)
                if d < or_ * 1.2:
                    node_penalty += 500 + (or_ * 1.2 - d) * 200

            # penalty 2: overlap with already-placed label bboxes
            label_penalty = 0
            for lb in placed_bboxes:
                lx0, ly0, lx1, ly1 = lb
                ox_ov = max(0, min(bx1, lx1) - max(bx0, lx0))
                oy_ov = max(0, min(by1, ly1) - max(by0, ly0))
                if ox_ov > 0 and oy_ov > 0:
                    label_penalty += 800 + ox_ov * oy_ov * 3000

            # penalty 3: prefer directions away from graph centre
            cx_graph = np.mean([pos[n][0] for n in nodes])
            cy_graph = np.mean([pos[n][1] for n in nodes])
            # positive = label moves away from centre (good)
            outward  = (udx * (x - cx_graph) + udy * (y - cy_graph))
            outward_penalty = -outward * 0.5   # small reward for outward

            score = node_penalty + label_penalty + outward_penalty

            if score < best_score:
                best_score = score
                best_pos   = (tx, ty, bx0, by0, bx1, by1)

        tx, ty, bx0, by0, bx1, by1 = best_pos
        placements[node] = (tx, ty)
        placed_bboxes.append((bx0, by0, bx1, by1))

    return placements


def draw_network(G, pos, freq_map, out_path):
    deg_cen      = nx.degree_centrality(G)
    sorted_nodes = sorted(G.nodes(), key=lambda x: deg_cen[x], reverse=True)
    nodes_list   = list(G.nodes())

    sizes  = [nsize(n, sorted_nodes, deg_cen, freq_map) for n in nodes_list]
    colors = [ncolor(tier(n, sorted_nodes)) for n in nodes_list]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor(PAL["bg"])

    # draw edges
    for edge in G.edges():
        w = G.edges[edge]["weight"]
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
                               width=0.8 + 1.5*w,
                               alpha=min(0.12 + 0.07*w, 0.55),
                               edge_color=PAL["grey"], ax=ax)

    # draw nodes (halo + fill)
    ax.scatter([pos[n][0] for n in nodes_list],
               [pos[n][1] for n in nodes_list],
               s=[s*1.5 for s in sizes], c=colors, alpha=0.10,
               zorder=3, linewidths=0)
    ax.scatter([pos[n][0] for n in nodes_list],
               [pos[n][1] for n in nodes_list],
               s=sizes, c=colors, edgecolors="white",
               linewidth=1.8, alpha=0.93, zorder=4)

    # need axes to be "drawn" so we can do coord conversions
    fig.canvas.draw()

    # ── compute centrality-box footprint in data coords so labels avoid it ──
    # The box sits at axes (0.99, 0.98) top-right. Estimate its size:
    # 7 lines of text at fontsize 10
    cbox_lines = 7
    cbox_chars = 38   # rough max chars per line
    pt_w = cbox_chars * 10 * 0.55
    pt_h = cbox_lines * 10 * 1.35
    disp_per_pt = fig.dpi / 72.0
    ax_win   = ax.get_window_extent()
    xlim     = ax.get_xlim()
    ylim     = ax.get_ylim()
    ax_dw    = xlim[1] - xlim[0]
    ax_dh    = ylim[1] - ylim[0]
    cbox_w_d = pt_w * disp_per_pt / ax_win.width  * ax_dw
    cbox_h_d = pt_h * disp_per_pt / ax_win.height * ax_dh
    # axes (0.99, 0.98) top-right → data coords of top-right corner
    cbox_x1  = xlim[0] + 0.99 * ax_dw
    cbox_y1  = ylim[0] + 0.98 * ax_dh
    cbox_x0  = cbox_x1 - cbox_w_d
    cbox_y0  = cbox_y1 - cbox_h_d
    # add generous padding so labels stay well clear
    pad_d = 0.04 * ax_dw
    reserved = [(cbox_x0 - pad_d, cbox_y0 - pad_d,
                 cbox_x1 + pad_d, cbox_y1 + pad_d)]

    # compute smart label positions
    placements = place_labels(nodes_list, pos, sizes, sorted_nodes,
                               deg_cen, freq_map, ax, fig,
                               reserved_bboxes=reserved)

    # draw labels with leader lines
    for node in nodes_list:
        node_x, node_y = pos[node]
        tx, ty = placements[node]
        t      = tier(node, sorted_nodes)
        col    = ncolor(t)
        fsize  = 13 if t in ("top1", "top") else 11
        fw     = "bold" if t in ("top1", "top") else "normal"

        # leader line from node edge toward label
        s_pt2  = sizes[nodes_list.index(node)]
        r_data = node_radius_data(s_pt2, ax, fig)
        dx, dy = tx - node_x, ty - node_y
        dist   = math.hypot(dx, dy)
        if dist > 0:
            # start line at node circumference
            lx = node_x + dx / dist * r_data * 1.05
            ly = node_y + dy / dist * r_data * 1.05
            ax.plot([lx, tx], [ly, ty], color=col, lw=0.6,
                    alpha=0.5, zorder=5)

        ax.annotate(node, xy=(tx, ty),
                    ha="center", va="center",
                    fontsize=fsize, fontweight=fw, color=PAL["dark"],
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=col, lw=0.9, alpha=0.95),
                    zorder=6, annotation_clip=False)

    # centrality box
    top6  = sorted_nodes[:6]
    lines = ["Top keywords by centrality:"]
    for i, nd in enumerate(top6, 1):
        lines.append(f"  {i}. {nd}  (deg={deg_cen[nd]:.2f})")
    ax.text(0.99, 0.98, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=PAL["dark"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.90))

    # legend
    leg = [
        mpatches.Patch(color=PAL["orange"], label="Highest centrality (Top 1)"),
        mpatches.Patch(color=PAL["teal"],   label="High centrality (Top 2–4)"),
        mpatches.Patch(color=PAL["blue"],   label="Medium centrality"),
        mpatches.Patch(color=PAL["purple"], label="Other keywords"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=9,
              frameon=True, edgecolor="#CCCCCC", framealpha=0.92)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


# ── Figure 7a ──────────────────────────────────────────────────────────────
papers_star = parse_papers(df)
freq_star   = Counter(itertools.chain.from_iterable(papers_star))
G_star      = build_graph(papers_star, freq_star, TOP_N, 2)
n           = G_star.number_of_nodes()
pos_star    = nx.spring_layout(G_star, seed=SEED, k=2.2/math.sqrt(n), iterations=500)
draw_network(G_star, pos_star, freq_star,
             "/mnt/user-data/outputs/Figure7a_v3.png")

# ── Figure 7b ──────────────────────────────────────────────────────────────
papers_core = parse_papers(df, extra_stop={"food waste", "consumer food waste"})
freq_core   = Counter(itertools.chain.from_iterable(papers_core))
G_core      = build_graph(papers_core, freq_core, 18, 1)
pos_core    = nx.spring_layout(G_core, seed=SEED, k=2.2, iterations=500, weight="weight")
draw_network(G_core, pos_core, freq_core,
             "/mnt/user-data/outputs/Figure7b_v3.png")
