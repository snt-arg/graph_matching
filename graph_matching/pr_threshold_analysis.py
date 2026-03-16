#!/usr/bin/env python3
"""
Precision-Recall threshold analysis for Graph Matching.

Sweeps a confidence threshold over match scores and shows:
  - Precision-Recall curve (where you are now vs where you could be)
  - Score distribution of TP vs FP matches (how separable the two populations are)
  - Optimal F1 threshold

Works with both matchers:
  Classic matcher (default)  : uses CLIPPER scores from GraphMatcher.match()
  GNN matcher   (--use-gnn)  : uses soft Sinkhorn scores from PGM_class.py

Usage:
    python pr_threshold_analysis.py              # classic matcher
    python pr_threshold_analysis.py --use-gnn    # GNN matcher
    python pr_threshold_analysis.py --both       # compare side by side
    python pr_threshold_analysis.py --save fig.png
"""

import os
import sys
import copy
import json
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Shared imports from Script 1 ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matching_synthetic_dataset import (
    compute_metrics,
    process_synthetic_graph,
    syntheticDS_params,
    fake_logger,
)

# ── GNN path ──────────────────────────────────────────────────────────────────
GNN_CLASS_PATH = '/root/workspace/src/graph_matching_gnn/graph_matching/graph_matching'
GNN_DATA_PATH  = '/root/workspace/src/graph_matching_gnn/GNN'

# ── Dataset path ──────────────────────────────────────────────────────────────
PICKLE_PATH = "/home/adminpc/datasets_matching/pickles/incremental_translation.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Core PR utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_pr_curve(tp_scores, fp_scores, fn_count, thresholds):
    """
    Sweep thresholds and return precision / recall / F1 arrays.

    At threshold τ a predicted pair is accepted only if its score >= τ.
    FN at τ = (irreducible FN from fn_count) + (TP pairs filtered out by τ).

    Returns
    -------
    precisions, recalls, f1s : np.ndarray, shape (len(thresholds),)
    """
    all_scores = np.array(tp_scores + fp_scores)
    all_labels = np.array([1] * len(tp_scores) + [0] * len(fp_scores))

    precisions, recalls, f1s = [], [], []
    for tau in thresholds:
        accepted = all_scores >= tau
        tp = int((accepted & (all_labels == 1)).sum())
        fp = int((accepted & (all_labels == 0)).sum())
        fn = fn_count + int((~accepted & (all_labels == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.array(precisions), np.array(recalls), np.array(f1s)


# ─────────────────────────────────────────────────────────────────────────────
# Classic matcher: score collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_scores_classic(full_dataset):
    """
    Run GraphMatcher on every pair in the dataset.
    For each unambiguous (single-solution) result collect per-pair
    (score, is_tp / is_fp) data.  Pairs in GT never returned are FN.

    The CLIPPER score on each pair is the consistency score of the local
    group — higher means more geometrically confident.

    Returns
    -------
    tp_scores : list[float]
    fp_scores : list[float]
    fn_count  : int
    """
    from GraphMatcher import GraphMatcher

    tp_scores, fp_scores = [], []
    fn_count = 0

    for a_graph, s_graphs in tqdm(full_dataset, desc="Classic matcher", colour="blue"):
        process_synthetic_graph(a_graph)
        for s_graph in s_graphs:
            process_synthetic_graph(s_graph)
            a_graph.name = "A-Graph"
            s_graph.name = "S-Graph"
            a_graph.stringify_node_ids()
            s_graph.stringify_node_ids()

            # Ground truth: every S-graph node ID should map to itself in A-graph
            s_graph_trimmed = copy.deepcopy(s_graph).filter_graph_by_node_types(["room", "ws"])
            gt_set = {(str(i), str(i)) for i in s_graph_trimmed.get_nodes_ids()}

            graph_matcher = GraphMatcher(fake_logger, log_level=0)
            graph_matcher.set_parameters(syntheticDS_params)
            graph_matcher.set_graph_from_wrapper(a_graph, "A-Graph")
            graph_matcher.set_graph_from_wrapper(s_graph, "S-Graph")
            graph_matcher.room_string = "room"
            graph_matcher.ws_string   = "ws"

            success, matches, _, _ = graph_matcher.match("A-Graph", "S-Graph")

            if not success or not matches:
                fn_count += len(gt_set)
                continue

            # Skip ambiguous cases (symmetry) — consistent with compute_metrics usage
            if len(matches) != 1:
                fn_count += len(gt_set)
                continue

            proposed_pairs = set()
            for pair in matches[0]:
                node_a = str(pair['origin_node'])
                node_b = str(pair['target_node'])
                score  = float(pair['score'])
                proposed_pairs.add((node_a, node_b))
                if (node_a, node_b) in gt_set:
                    tp_scores.append(score)
                else:
                    fp_scores.append(score)

            fn_count += len(gt_set - proposed_pairs)

    return tp_scores, fp_scores, fn_count


# ─────────────────────────────────────────────────────────────────────────────
# GNN matcher: score collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_scores_gnn():
    """
    Load the GNN model with its own test set and run soft inference
    (discrete=False) to collect per-match confidence scores.

    Uses collect_match_scores() from PGM_class.py so the logic is
    identical to what PartialGraphMatching.pr_analysis() uses.

    Returns
    -------
    tp_scores : list[float]
    fp_scores : list[float]
    fn_count  : int
    """
    import torch

    if GNN_CLASS_PATH not in sys.path:
        sys.path.insert(0, GNN_CLASS_PATH)

    from PGM_class import (
        PartialGraphMatching,
        MatchingModel_GATv2SinkhornTopK,
        collect_match_scores,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pgm = PartialGraphMatching(
        model_class=MatchingModel_GATv2SinkhornTopK,
        data_paths={
            "equal":   os.path.join(GNN_DATA_PATH, "preprocessed", "graph_matching", "equal"),
            "partial": os.path.join(GNN_DATA_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise"),
        },
        model_save_path=os.path.join(GNN_DATA_PATH, "models", "partial_graph_matching", "ws_room_dropout_noise"),
        device=device,
        in_dim=7,
    )
    pgm.load_best_model()

    return collect_match_scores(pgm.model, pgm.test_list)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(results_dict, thresholds=None, save_path=None):
    """
    Plot one row per matcher: PR curve (left) + score distribution (right).

    Parameters
    ----------
    results_dict : {label: (tp_scores, fp_scores, fn_count)}
    thresholds   : 1-D array (default: 200 values in [0, 1])
    save_path    : if given, saves the figure
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 200)

    n_rows = len(results_dict)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, (label, (tp_scores, fp_scores, fn_count)) in enumerate(results_dict.items()):
        precisions, recalls, f1s = compute_pr_curve(
            tp_scores, fp_scores, fn_count, thresholds)

        best_idx = int(np.argmax(f1s))
        best_tau = thresholds[best_idx]
        best_f1  = f1s[best_idx]
        best_p   = precisions[best_idx]
        best_r   = recalls[best_idx]

        print(f"\n[{label}]")
        print(f"  Optimal τ = {best_tau:.4f}  →  "
              f"P={best_p:.4f}  R={best_r:.4f}  F1={best_f1:.4f}")
        print(f"  Baseline (τ=0): "
              f"P={precisions[0]:.4f}  R={recalls[0]:.4f}  F1={f1s[0]:.4f}")

        ax1, ax2 = axes[row]

        # ── PR curve ─────────────────────────────────────────────────────────
        ax1.plot(recalls, precisions, lw=2, label="PR curve")
        ax1.scatter([best_r], [best_p], color="red", zorder=5, s=80,
                    label=f"Optimal F1={best_f1:.3f}  (τ={best_tau:.3f})")
        ax1.scatter([recalls[0]], [precisions[0]], marker="x", color="gray",
                    zorder=5, s=120, linewidths=2,
                    label=f"No threshold (τ=0)\nP={precisions[0]:.3f}  R={recalls[0]:.3f}")
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.set_title(f"{label} — Precision-Recall Curve")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])
        ax1.legend(fontsize=9)
        ax1.grid(True)

        # ── Score distributions ───────────────────────────────────────────────
        bins = np.linspace(0, 1, 40)
        ax2.hist(tp_scores, bins=bins, alpha=0.6, color="green",
                 label=f"TP  (n={len(tp_scores)})")
        ax2.hist(fp_scores, bins=bins, alpha=0.6, color="red",
                 label=f"FP  (n={len(fp_scores)})")
        ax2.axvline(best_tau, color="black", linestyle="--", lw=1.5,
                    label=f"Optimal τ={best_tau:.3f}")
        ax2.set_xlabel("Confidence score")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{label} — Score Distribution: TP vs FP")
        ax2.legend(fontsize=9)
        ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PR threshold analysis — classic or GNN graph matcher")
    parser.add_argument("--use-gnn", action="store_true",
                        help="Analyze GNN matcher (default: classic)")
    parser.add_argument("--both",    action="store_true",
                        help="Analyze both matchers side by side")
    parser.add_argument("--save",    type=str, default=None,
                        help="Save figure to this path instead of displaying")
    args = parser.parse_args()

    run_classic = not args.use_gnn or args.both
    run_gnn     = args.use_gnn     or args.both

    results_dict = {}

    if run_classic:
        print("Loading dataset for classic matcher...")
        full_dataset = pickle.load(open(PICKLE_PATH, "rb"))
        tp, fp, fn = collect_scores_classic(full_dataset)
        results_dict["Classic Matcher"] = (tp, fp, fn)

    if run_gnn:
        print("Loading GNN model and test set...")
        tp, fp, fn = collect_scores_gnn()
        results_dict["GNN Matcher"] = (tp, fp, fn)

    plot_pr_curves(results_dict, save_path=args.save)


if __name__ == "__main__":
    main()
