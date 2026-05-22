#!/usr/bin/env python3
"""
Standalone visualization of a single BA (m=3, N=150) network.
Nodes are sized and colored by betweenness centrality.
Edge alpha is scaled by the average BC of its endpoints.
Outputs: figures/network_ba_m3.pdf  and  figures/network_ba_m3.png
"""
import sys, os, random
import numpy as np
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

sys.path.insert(0, '.')
from src.networks import make_ba

SEED = 7
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

random.seed(SEED)
ig.set_random_number_generator(random.Random(SEED))
g = make_ba(n=150, m=3, seed=SEED)

# ── centralities ──────────────────────────────────────────────────────────────
bc = np.array(g.betweenness(directed=False))
try:
    ec = np.array(g.eigenvector_centrality(directed=False, scale=True))
except ig.InternalError:
    ec = np.ones(g.vcount())

bc_norm = (bc - bc.min()) / (bc.max() - bc.min() + 1e-12)
ec_norm = (ec - ec.min()) / (ec.max() - ec.min() + 1e-12)

# ── layout ────────────────────────────────────────────────────────────────────
layout = np.array(g.layout('fr').coords)

# ── figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.8))
fig.patch.set_facecolor('#0d1117')

panels = [
    (bc_norm, 'Betweenness Centrality', 'plasma'),
    (ec_norm, 'Eigenvector Centrality', 'viridis'),
]

for col, (scores, title, cmap_name) in enumerate(panels):
    ax = fig.add_axes([0.04 + col * 0.48, 0.06, 0.42, 0.76])
    ax.set_facecolor('#0d1117')

    cmap = plt.get_cmap(cmap_name)

    # edges — draw low-score edges first (behind)
    edge_scores = np.array([
        (scores[e.source] + scores[e.target]) / 2 for e in g.es
    ])
    order = np.argsort(edge_scores)
    for i in order:
        e = g.es[i]
        s, t = e.source, e.target
        alpha = 0.12 + 0.55 * edge_scores[i]
        lw    = 0.4  + 1.0  * edge_scores[i]
        ax.plot(
            [layout[s, 0], layout[t, 0]],
            [layout[s, 1], layout[t, 1]],
            color='white', alpha=alpha, lw=lw, solid_capstyle='round', zorder=1,
        )

    # nodes — draw low-score nodes first
    node_order = np.argsort(scores)
    node_sizes = 18 + 320 * scores ** 1.4
    sc = ax.scatter(
        layout[node_order, 0], layout[node_order, 1],
        c=scores[node_order], cmap=cmap_name,
        s=node_sizes[node_order],
        vmin=0, vmax=1,
        linewidths=0.6,
        edgecolors=np.array([cmap(v) for v in scores[node_order]]) * 0.6 + 0.4,
        zorder=2,
    )

    ax.set_title(title, color='white', fontsize=13, pad=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # colorbar
    cbar_ax = fig.add_axes([0.04 + col * 0.48 + 0.42, 0.14, 0.012, 0.56])
    cb = ColorbarBase(cbar_ax, cmap=cmap_name,
                      norm=mcolors.Normalize(0, 1), orientation='vertical')
    cb.set_label('Normalized score', color='#aaaaaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')
    cb.outline.set_edgecolor('#333333')

# ── shared title ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.97,
         'Barabási–Albert network  (N=150, m=3)',
         ha='center', va='top', color='white', fontsize=15, fontweight='bold')
fig.text(0.50, 0.925,
         f'{g.vcount()} nodes · {g.ecount()} edges · '
         f'⟨k⟩ = {2*g.ecount()/g.vcount():.1f}',
         ha='center', va='top', color='#888888', fontsize=10)

out_pdf = f'{FIG_DIR}/network_ba_m3.pdf'
out_png = f'{FIG_DIR}/network_ba_m3.png'
fig.savefig(out_pdf, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
fig.savefig(out_png, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=200)
plt.close(fig)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')
