"""
Plot TP vs FP Sinkhorn score distributions for 47_basement.

Two subplots:
  Left  — full Sinkhorn matrix: all GT-match cells (TP) vs all GT-non-match cells (FP)
  Right — Hungarian-selected cells only: correct assignments (TP) vs wrong assignments (FP)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ────────────────────────────────────────────────────────────────────
PGM_PATH = '/root/workspace/src/graph_matching_gnn/graph_matching'
GNN_PATH  = '/root/workspace/src/graph_matching_gnn/GNN'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if PGM_PATH not in sys.path:
    sys.path.insert(0, PGM_PATH)

# reuse helpers from matching_synthetic_dataset
sys.path.insert(0, SCRIPT_DIR)

import torch
from PGM_class import (
    PartialGraphMatching, MatchingModel_GATv2SinkhornTopK,
    predict_matching_matrix, nx_to_pyg_data_preserve_order, normalize_graph,
)

# import load_real_graphs and related helpers
from matching_synthetic_dataset import load_real_graphs

# ── load model ───────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(GNN_PATH, "models", "partial_graph_matching",
                          "ws_room_dropout_noise_inc_BCE")

pgm = PartialGraphMatching(
    model_class=MatchingModel_GATv2SinkhornTopK,
    data_paths={
        "equal":   os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal"),
        "partial": os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching",
                                "ws_room_dropout_noise_inc"),
    },
    model_save_path=model_path,
    device=device,
    in_dim=7,
)
pgm.load_best_model()
print("PGM model loaded.")

# ── load 47_basement graphs & ground truth ───────────────────────────────────
a_graph, s_graph, gt_match = load_real_graphs("47_basement")

# remove isolated nodes
for label, g in (("Prior", a_graph), ("Online", s_graph)):
    isolated = [n for n in g.graph.nodes() if g.graph.degree(n) == 0]
    if isolated:
        g.graph.remove_nodes_from(isolated)
        print(f"[47_basement] {label}: removed {len(isolated)} isolated nodes "
              f"({g.graph.number_of_nodes()} remaining)")

g1 = a_graph.graph
g2 = s_graph.graph

g1_nodes = list(g1.nodes())
g2_nodes = list(g2.nodes())
N1, N2 = len(g1_nodes), len(g2_nodes)
print(f"Prior nodes: {N1}, Online nodes: {N2}, GT pairs: {len(gt_match)}")

# ── run inference — soft matrix & hard matrix ────────────────────────────────
g1_pyg = nx_to_pyg_data_preserve_order(g1)
g2_pyg = nx_to_pyg_data_preserve_order(g2)
g1_pyg, g2_pyg = normalize_graph(g1_pyg, g2_pyg, pgm.mean, pgm.std)

soft_S = predict_matching_matrix(pgm.model, g1_pyg, g2_pyg, discrete=False).cpu().numpy()
hard_S = predict_matching_matrix(pgm.model, g1_pyg, g2_pyg, discrete=True).cpu().numpy()

print(f"Soft matrix shape: {soft_S.shape}, Hard matrix shape: {hard_S.shape}")

# ── build ground-truth mask ──────────────────────────────────────────────────
g1_idx = {str(n): i for i, n in enumerate(g1_nodes)}
g2_idx = {str(n): i for i, n in enumerate(g2_nodes)}

gt_mask = np.zeros((N1, N2), dtype=float)
missing = []
for prior_id, online_id in gt_match:
    i = g1_idx.get(str(prior_id))
    j = g2_idx.get(str(online_id))
    if i is not None and j is not None:
        gt_mask[i, j] = 1.0
    else:
        missing.append((prior_id, online_id))

if missing:
    print(f"[WARN] {len(missing)} GT pairs not found in graph nodes: {missing[:5]}")

n_gt = int(gt_mask.sum())
print(f"GT mask: {n_gt} true pairs")

# ── separate scores ──────────────────────────────────────────────────────────
# Full Sinkhorn matrix
full_tp = soft_S[gt_mask == 1].flatten()
full_fp = soft_S[gt_mask == 0].flatten()

# Hungarian-selected cells only
hard_tp = soft_S[(hard_S > 0) & (gt_mask == 1)].flatten()
hard_fp = soft_S[(hard_S > 0) & (gt_mask == 0)].flatten()

print(f"Hungarian selected: {int(hard_S.sum())} pairs — TP={len(hard_tp)}, FP={len(hard_fp)}")

# ── plot ─────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(13, 5))
fig.suptitle("47_basement — Sinkhorn Score Distributions: TP vs FP", fontsize=13, y=1.01)

gs = gridspec.GridSpec(1, 2, wspace=0.35)
ax_full = fig.add_subplot(gs[0])
ax_hard = fig.add_subplot(gs[1])

colors = {'tp': '#2ca02c', 'fp': '#d62728'}
x_range = np.linspace(0, 1, 500)

def _plot_kde(ax, arr, color, label):
    if len(arr) == 0:
        return
    if len(arr) == 1:
        ax.axvline(arr[0], color=color, linewidth=2, linestyle='--', label=label)
        return
    kde = gaussian_kde(arr)
    y = kde(x_range)
    ax.plot(x_range, y, color=color, linewidth=2, label=label)
    ax.fill_between(x_range, y, alpha=0.15, color=color)

# Left: full Sinkhorn matrix
_plot_kde(ax_full, full_tp, colors['tp'], f"GT match (TP)  n={len(full_tp):,}")
_plot_kde(ax_full, full_fp, colors['fp'], f"GT non-match (FP)  n={len(full_fp):,}")
ax_full.set_xlim(0, 1)
ax_full.set_xlabel("Sinkhorn score")
ax_full.set_ylabel("Density")
ax_full.set_title("All Sinkhorn entries\n(full matrix)")
ax_full.legend(fontsize=9)
ax_full.grid(True, alpha=0.3)

# Right: Hungarian-selected cells
_plot_kde(ax_hard, hard_tp, colors['tp'], f"Correct assignment (TP)  n={len(hard_tp)}")
_plot_kde(ax_hard, hard_fp, colors['fp'], f"Wrong assignment (FP)   n={len(hard_fp)}")
ax_hard.set_xlim(0, 1)
ax_hard.set_xlabel("Sinkhorn score")
ax_hard.set_ylabel("Density")
ax_hard.set_title("Hungarian-selected pairs only\n(final assignments)")
ax_hard.legend(fontsize=9)
ax_hard.grid(True, alpha=0.3)

fig.tight_layout()

# save
results_dir = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "47_basement_tp_fp_distribution.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.show()
