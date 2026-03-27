#!/usr/bin/env python3
"""
Graph Matching Results Plotter

Reads JSON results from graph matching experiments and creates
comprehensive visualizations of performance metrics.

Usage:
    python plot_results.py [results_file.json]

If no file is specified, looks for 'latest_results.json' in the same directory.
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
from collections import defaultdict
import datetime


def load_results(json_filepath):
    """Load experiment results from JSON file."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {json_filepath} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_filepath}")
        return None


def process_data(data):
    """Process raw data and group by number of rooms."""
    experiments = data['experiments']
    metadata = data['metadata']

    metrics_by_rooms = defaultdict(list)
    success_data     = defaultdict(list)
    timing_data      = defaultdict(list)
    solution_count_data = defaultdict(list)
    raw_points = []

    _cls_metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']

    for experiment in experiments:
        n_rooms_s = experiment['n_rooms_s_graphs']
        n_rooms_a = experiment.get('n_rooms_a_graphs', n_rooms_s)
        matching_time = experiment.get('matching_time', 0.0)
        num_solutions = experiment.get('num_solutions', 0)

        timing_data[n_rooms_s].append(matching_time)
        solution_count_data[n_rooms_s].append(num_solutions)

        is_success = experiment.get('success', bool(experiment.get('metrics')))
        success_data[n_rooms_s].append(1 if is_success else 0)

        has_metrics = ('metrics' in experiment
                       and bool(experiment['metrics'])
                       and 'precision' in experiment['metrics'])
        if has_metrics:
            metrics_by_rooms[n_rooms_s].append(experiment['metrics'])

        point = {
            'n_rooms_s':    n_rooms_s,
            'n_rooms_a':    n_rooms_a,
            'success':      1 if is_success else 0,
            'time':         matching_time,
            'num_solutions': num_solutions,
        }
        for m in _cls_metrics:
            point[m] = (experiment['metrics'][m]
                        if has_metrics and m in experiment['metrics']
                        else np.nan)
        raw_points.append(point)

    return metrics_by_rooms, success_data, timing_data, solution_count_data, metadata, raw_points


def calculate_statistics(metrics_by_rooms, success_data, timing_data, solution_count_data):
    """Calculate per-room-count statistics."""
    all_room_counts = set(metrics_by_rooms) | set(success_data) | set(timing_data)
    room_counts = sorted(all_room_counts)

    metric_names  = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']

    averaged_metrics = {m: [] for m in metric_names}
    std_metrics      = {m: [] for m in metric_names}
    success_rates, avg_times, std_times, avg_solutions, std_solutions = [], [], [], [], []

    for rc in room_counts:
        # Classification metrics
        if rc in metrics_by_rooms:
            ml = metrics_by_rooms[rc]
            for m in metric_names:
                vals = [x[m] for x in ml]
                averaged_metrics[m].append(np.mean(vals))
                std_metrics[m].append(np.std(vals))
        else:
            for m in metric_names:
                averaged_metrics[m].append(np.nan)
                std_metrics[m].append(np.nan)

        # Success rates
        if rc in success_data:
            success_rates.append(np.mean(success_data[rc]) * 100)
        else:
            success_rates.append(np.nan)

        # Timing
        if rc in timing_data:
            avg_times.append(np.mean(timing_data[rc]))
            std_times.append(np.std(timing_data[rc]))
        else:
            avg_times.append(np.nan)
            std_times.append(np.nan)

        # Solution counts
        if rc in solution_count_data:
            avg_solutions.append(np.mean(solution_count_data[rc]))
            std_solutions.append(np.std(solution_count_data[rc]))
        else:
            avg_solutions.append(np.nan)
            std_solutions.append(np.nan)

    stats = {
        'averaged_metrics': averaged_metrics,
        'std_metrics':      std_metrics,
        'success_rates':    success_rates,
        'avg_times':        avg_times,
        'std_times':        std_times,
        'avg_solutions':    avg_solutions,
        'std_solutions':    std_solutions,
    }
    return room_counts, metric_names, metric_labels, stats


def _attach_expand_on_dblclick(fig):
    """Double-click a subplot to expand it; double-click again to restore."""
    axes_list = fig.get_axes()
    orig_pos  = {ax: ax.get_position() for ax in axes_list}
    orig_vis  = {ax: ax.get_visible()  for ax in axes_list}
    state = [None]

    def on_dblclick(event):
        if not event.dblclick:
            return
        dpi = fig.dpi
        fx  = event.x / (fig.get_figwidth()  * dpi)
        fy  = event.y / (fig.get_figheight() * dpi)

        if state[0] is not None:
            for ax in axes_list:
                ax.set_position(orig_pos[ax])
                ax.set_visible(orig_vis[ax])
            state[0] = None
        else:
            clicked = next(
                (ax for ax in axes_list
                 if orig_pos[ax].x0 <= fx <= orig_pos[ax].x1
                 and orig_pos[ax].y0 <= fy <= orig_pos[ax].y1),
                None
            )
            if clicked is None:
                return
            for ax in axes_list:
                if ax is clicked:
                    ax.set_position([0.05, 0.05, 0.9, 0.9])
                else:
                    ax.set_visible(False)
            state[0] = clicked

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_dblclick)


def create_3d_plots(raw_points, metadata, save_path=None, show_scatter=True,
                    threshold_points=None):
    """3D surface plots: X = n_rooms_s, Y = n_rooms_a, Z = metric.

    If *threshold_points* is provided (list of dicts with keys 'threshold',
    'accuracy', 'recall'), a 9th 3D subplot is added:
        X = score_threshold, Y = accuracy, Z = recall.
    The grid is extended from 3×3 to 3×4 to accommodate it.
    """
    plot_specs = [
        ('precision',     'Precision'),
        ('recall',        'Recall'),
        ('f1_score',      'F1-Score'),
        ('accuracy',      'Accuracy'),
        ('specificity',   'Specificity'),
        ('success',       'Success (0/1)'),
        ('time',          'Matching Time (s)'),
        ('num_solutions', 'Num Solutions'),
    ]
    NUM_SOLUTIONS_THRESHOLD = 1.5

    has_threshold = bool(threshold_points)
    ncols = 4 if has_threshold else 3
    figw  = 36 if has_threshold else 28

    fig = plt.figure(figsize=(figw, 20))
    fig.suptitle('Graph Matching: 3D View  (S-Rooms × A-Rooms)', fontsize=16)

    for plot_idx, (field, zlabel) in enumerate(plot_specs):
        ax = fig.add_subplot(3, ncols, plot_idx + 1, projection='3d')

        xs = np.array([p['n_rooms_s'] for p in raw_points], dtype=float)
        ys = np.array([p['n_rooms_a'] for p in raw_points], dtype=float)
        zs = np.array([p[field]       for p in raw_points], dtype=float)

        valid = ~np.isnan(zs)
        xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

        if len(xs_v) > 0:
            unique_xy = np.unique(np.column_stack([xs_v, ys_v]), axis=0)
            if (len(unique_xy) >= 6
                    and len(np.unique(xs_v)) >= 2
                    and len(np.unique(ys_v)) >= 2):
                xi = np.linspace(xs_v.min(), xs_v.max(), 30)
                yi = np.linspace(ys_v.min(), ys_v.max(), 30)
                Xi, Yi = np.meshgrid(xi, yi)
                try:
                    xy_to_zs = defaultdict(list)
                    for x, y, z in zip(xs_v, ys_v, zs_v):
                        xy_to_zs[(x, y)].append(z)
                    xs_agg = np.array([k[0] for k in xy_to_zs])
                    ys_agg = np.array([k[1] for k in xy_to_zs])
                    zs_agg = np.array([np.mean(v) for v in xy_to_zs.values()])
                    Zi = griddata((xs_agg, ys_agg), zs_agg, (Xi, Yi), method='linear')

                    if field == 'num_solutions':
                        import matplotlib.colors as mcolors
                        norm_max   = max(zs_agg.max(), NUM_SOLUTIONS_THRESHOLD + 0.01)
                        cmap_nodes = [0.0, NUM_SOLUTIONS_THRESHOLD, NUM_SOLUTIONS_THRESHOLD, norm_max]
                        norm_nodes = [v / norm_max for v in cmap_nodes]
                        cmap = mcolors.LinearSegmentedColormap.from_list(
                            'thresh',
                            list(zip(norm_nodes, ['#d62728', '#d62728', '#1f77b4', '#1f77b4'])))
                        z_norm = mcolors.Normalize(vmin=0, vmax=norm_max)
                        ax.plot_surface(Xi, Yi, Zi, facecolors=cmap(z_norm(Zi)),
                                        alpha=0.5, linewidth=0, antialiased=True)
                    else:
                        ax.plot_surface(Xi, Yi, Zi, alpha=0.35, color='#1f77b4',
                                        linewidth=0, antialiased=True)
                except Exception:
                    pass

            if show_scatter:
                ax.scatter(xs_v, ys_v, zs_v, color='#1f77b4', s=20, depthshade=True)

        ax.set_xlabel('S-Rooms', fontsize=8, labelpad=3)
        ax.set_ylabel('A-Rooms', fontsize=8, labelpad=3)
        ax.set_zlabel(zlabel,    fontsize=8, labelpad=3)
        ax.set_title(zlabel, fontsize=10, fontweight='bold')

    # ── Threshold subplot (slot 9 in the 3×4 grid) ───────────────────────────
    if has_threshold:
        ax_t = fig.add_subplot(3, ncols, 9, projection='3d')

        xs_t = np.array([p['threshold'] for p in threshold_points], dtype=float)
        ys_t = np.array([p['accuracy']  for p in threshold_points], dtype=float)
        zs_t = np.array([p['recall']    for p in threshold_points], dtype=float)

        valid_t = ~(np.isnan(xs_t) | np.isnan(ys_t) | np.isnan(zs_t))
        xs_tv, ys_tv, zs_tv = xs_t[valid_t], ys_t[valid_t], zs_t[valid_t]

        if len(xs_tv) > 0:
            unique_xy_t = np.unique(np.column_stack([xs_tv, ys_tv]), axis=0)
            if (len(unique_xy_t) >= 6
                    and len(np.unique(xs_tv)) >= 2
                    and len(np.unique(ys_tv)) >= 2):
                xi_t = np.linspace(xs_tv.min(), xs_tv.max(), 30)
                yi_t = np.linspace(ys_tv.min(), ys_tv.max(), 30)
                Xi_t, Yi_t = np.meshgrid(xi_t, yi_t)
                try:
                    xy_to_zs_t = defaultdict(list)
                    for x, y, z in zip(xs_tv, ys_tv, zs_tv):
                        xy_to_zs_t[(x, y)].append(z)
                    xs_agg_t = np.array([k[0] for k in xy_to_zs_t])
                    ys_agg_t = np.array([k[1] for k in xy_to_zs_t])
                    zs_agg_t = np.array([np.mean(v) for v in xy_to_zs_t.values()])
                    Zi_t = griddata((xs_agg_t, ys_agg_t), zs_agg_t, (Xi_t, Yi_t), method='linear')
                    ax_t.plot_surface(Xi_t, Yi_t, Zi_t, alpha=0.35, color='#d62728',
                                      linewidth=0, antialiased=True)
                except Exception:
                    pass

            ax_t.scatter(xs_tv, ys_tv, zs_tv, color='#d62728', s=20, depthshade=True)

        ax_t.set_xlabel('Threshold', fontsize=8, labelpad=3)
        ax_t.set_ylabel('Accuracy',  fontsize=8, labelpad=3)
        ax_t.set_zlabel('Recall',    fontsize=8, labelpad=3)
        ax_t.set_title('Recall vs Accuracy vs Threshold', fontsize=10, fontweight='bold')

    # ── Summary info (last cell) ──────────────────────────────────────────────
    summary_slot = 3 * ncols  # bottom-right cell
    ax_info = fig.add_subplot(3, ncols, summary_slot)
    ax_info.axis('off')
    ax_info.text(0.05, 0.95, 'Experiment Summary', transform=ax_info.transAxes,
                 fontsize=11, weight='bold', va='top')
    ax_info.text(0.05, 0.82, f"Total: {metadata['total_experiments']}",
                 transform=ax_info.transAxes, fontsize=9, va='top')
    ax_info.text(0.05, 0.70, f"Dataset:\n{metadata['dataset_file']}",
                 transform=ax_info.transAxes, fontsize=7, va='top')
    ax_info.text(0.05, 0.50, f"Generated:\n{metadata['timestamp'][:19]}",
                 transform=ax_info.transAxes, fontsize=9, va='top')

    plt.tight_layout()
    _attach_expand_on_dblclick(fig)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"3D plot saved to: {save_path}")

    return fig


def print_numerical_summary(room_counts, metric_names, metric_labels,
                            stats, metrics_by_rooms):
    """Print per-room-count statistics."""
    print("\n" + "="*80)
    print("SUMMARY: AVERAGED METRICS BY NUMBER OF ROOMS")
    print("="*80)

    for idx, rc in enumerate(room_counts):
        success_rate  = stats['success_rates'][idx]
        avg_time      = stats['avg_times'][idx]
        avg_solutions = stats['avg_solutions'][idx]

        print(f"\n  Rooms: {rc}")
        if not np.isnan(success_rate):
            print(f"    Success Rate  : {success_rate:.1f}%")
        if not np.isnan(avg_time):
            print(f"    Avg Time      : {avg_time:.4f}s ± {stats['std_times'][idx]:.4f}s")
        if not np.isnan(avg_solutions):
            print(f"    Avg Solutions : {avg_solutions:.2f} ± {stats['std_solutions'][idx]:.2f}")

        if rc in metrics_by_rooms:
            n = len(metrics_by_rooms[rc])
            print(f"    --- Classification metrics (N={n} unambiguous) ---")
            for m, label in zip(metric_names, metric_labels):
                mean_val = stats['averaged_metrics'][m][idx]
                std_val  = stats['std_metrics'][m][idx]
                if not np.isnan(mean_val):
                    print(f"    {label:12s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"    Classification metrics: N/A (no unambiguous solution)")


def save_summary_statistics(room_counts, metric_names, metric_labels,
                            stats, metrics_by_rooms, metadata, save_path):
    """Save detailed statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("GRAPH MATCHING PERFORMANCE ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {metadata['timestamp']}\n")
        f.write(f"Dataset:   {metadata['dataset_file']}\n")
        f.write(f"Total:     {metadata['total_experiments']} experiments\n\n")

        for idx, rc in enumerate(room_counts):
            success_rate  = stats['success_rates'][idx]
            avg_time      = stats['avg_times'][idx]
            avg_solutions = stats['avg_solutions'][idx]

            f.write(f"\nRooms: {rc}\n")
            if not np.isnan(success_rate):
                f.write(f"  Success Rate  : {success_rate:.1f}%\n")
            if not np.isnan(avg_time):
                f.write(f"  Avg Time      : {avg_time:.4f}s ± {stats['std_times'][idx]:.4f}s\n")
            if not np.isnan(avg_solutions):
                f.write(f"  Avg Solutions : {avg_solutions:.2f} ± {stats['std_solutions'][idx]:.2f}\n")

            if rc in metrics_by_rooms:
                n = len(metrics_by_rooms[rc])
                f.write(f"  --- Classification metrics (N={n} unambiguous) ---\n")
                for m, label in zip(metric_names, metric_labels):
                    mean_val = stats['averaged_metrics'][m][idx]
                    std_val  = stats['std_metrics'][m][idx]
                    if not np.isnan(mean_val):
                        f.write(f"  {label:12s}: {mean_val:.4f} ± {std_val:.4f}\n")
            else:
                f.write(f"  Classification metrics: N/A (no unambiguous solution)\n")

    print(f"Summary statistics saved to: {save_path}")


def collect_threshold_points(results_dir, current_dataset=None):
    """Scan result JSON files and return one dict per experiment with threshold info.

    Each returned dict has keys: 'threshold', 'accuracy', 'recall'.
    Files without 'score_threshold' in metadata are skipped.
    A None threshold is mapped to -0.05 so numpy can handle it as a float.
    """
    import glob

    raw_points = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, 'graph_matching_results_*.json'))):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        meta = data.get('metadata', {})
        if 'score_threshold' not in meta:
            continue

        threshold = meta['score_threshold']
        if current_dataset is not None and meta.get('dataset') != current_dataset:
            continue

        t_val = float(threshold) if threshold is not None else -0.05
        for exp in data.get('experiments', []):
            m = exp.get('metrics') or {}
            acc = m.get('accuracy')
            rec = m.get('recall')
            if acc is not None and rec is not None:
                raw_points.append({
                    'threshold': t_val,
                    'accuracy':  float(acc),
                    'recall':    float(rec),
                })

    return raw_points


def collect_soft_scores_from_msd(model, msd_test_list, sinkhorn_threshold=None, max_pairs=None,
                                  save_heatmaps_dir=None, heatmap_pairs=3):
    """
    Run the model on MSD pairs with return_soft=True to collect soft_topk scores
    labelled by whether hard_topk selected them AND whether they are true GT matches.
    Optionally saves matrix heatmaps for the first `heatmap_pairs` pairs.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    if save_heatmaps_dir:
        os.makedirs(save_heatmaps_dir, exist_ok=True)

    tp_scores, fp_scores, fn_scores, tn_scores = [], [], [], []
    aff_gt, aff_non_gt = [], []
    sk_gt,  sk_non_gt  = [], []
    sk_assigned_tp, sk_assigned_fp = [], []
    hung_gt, hung_non_gt = [], []

    pairs = msd_test_list if max_pairs is None else msd_test_list[:max_pairs]
    for pair_idx, (data1, data2, gt_perm) in enumerate(pairs):
        data1 = data1.to(device)
        data2 = data2.to(device)
        batch_idx1 = torch.zeros(data1.num_nodes, dtype=torch.long, device=device)
        batch_idx2 = torch.zeros(data2.num_nodes, dtype=torch.long, device=device)

        with torch.no_grad():
            hard_list, _, soft_list, affinity_list, sinkhorn_list = model(
                data1, data2, batch_idx1, batch_idx2,
                inference=True, return_soft=True,
                sinkhorn_threshold=sinkhorn_threshold,
                return_intermediate=True,
            )

        hard    = hard_list[0].cpu().numpy()      # [N1, N2] binary
        soft    = soft_list[0].cpu().numpy()      # [N1, N2] in [0, 1]
        affinity = affinity_list[0].cpu().numpy() # [N1, N2] normalised affinity
        sinkhorn = sinkhorn_list[0].cpu().numpy() # [N1, N2] doubly stochastic
        gt      = gt_perm.cpu().numpy()           # [N1, N2] binary ground truth

        # Resize gt to match soft if shapes differ (partial matching)
        N1, N2 = soft.shape
        if gt.shape != (N1, N2):
            gt_full = np.zeros((N1, N2), dtype=gt.dtype)
            r = min(gt.shape[0], N1)
            c = min(gt.shape[1], N2)
            gt_full[:r, :c] = gt[:r, :c]
            gt = gt_full

        tp_scores.append(soft[(hard == 1) & (gt == 1)].flatten())
        fp_scores.append(soft[(hard == 1) & (gt == 0)].flatten())
        fn_scores.append(soft[(hard == 0) & (gt == 1)].flatten())
        tn_scores.append(soft[(hard == 0) & (gt == 0)].flatten())

        aff_gt.append(affinity[gt == 1].flatten())
        aff_non_gt.append(affinity[gt == 0].flatten())
        sk_gt.append(sinkhorn[gt == 1].flatten())
        sk_non_gt.append(sinkhorn[gt == 0].flatten())
        hung_gt.append(hard[gt == 1].flatten())
        hung_non_gt.append(hard[gt == 0].flatten())
        # S scores at positions Hungarian assigned (hard=1), split by GT
        sk_assigned_tp.append(sinkhorn[(hard == 1) & (gt == 1)].flatten())
        sk_assigned_fp.append(sinkhorn[(hard == 1) & (gt == 0)].flatten())

        # if save_heatmaps_dir and pair_idx < heatmap_pairs:
        #     _save_single_heatmap(affinity, sinkhorn, soft, hard, gt,
        #                          pair_idx, save_heatmaps_dir)

    def _cat(lst): return np.concatenate(lst) if lst else np.array([])
    return (
        _cat(tp_scores), _cat(fp_scores), _cat(fn_scores), _cat(tn_scores),
        _cat(aff_gt), _cat(aff_non_gt),
        _cat(sk_gt),  _cat(sk_non_gt),
        _cat(hung_gt), _cat(hung_non_gt),
        _cat(sk_assigned_tp), _cat(sk_assigned_fp),
    )


def plot_soft_score_distribution(tp, fp, fn, tn):
    """
    Plot soft top-k score distributions split by hard selection and GT label.

    Four groups:
      TP : GT match  + hard selected   → want these high
      FP : GT non-match + hard selected → want these low (false alarms)
      FN : GT match  + NOT selected    → want these high (missed)
      TN : GT non-match + NOT selected → want these low (correct rejections)

    The overlap between FP and TP (or FN and TP) guides where to set a threshold.
    """
    from scipy.stats import gaussian_kde

    colors = {'TP': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e', 'TN': '#1f77b4'}
    x_range = np.linspace(0.0, 1.0, 500)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Soft top-k score distributions (selected vs filtered, by GT label)', fontsize=13)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # ── Row 0 left: selected pairs (hard=1) — TP vs FP ───────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for label, arr in [('TP', tp), ('FP', fp)]:
        if len(arr) > 1:
            kde = gaussian_kde(arr)
            ax.plot(x_range, kde(x_range), color=colors[label],
                    label=f'{label} (n={len(arr)})', linewidth=2)
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=colors[label])
    ax.set_xlabel('Soft top-k score  [0, 1]')
    ax.set_ylabel('Density')
    ax.set_title('Hard-selected pairs (hard=1)\nTP = correct match   FP = wrong match')
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 0 right: rejected pairs (hard=0) — FN vs TN ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for label, arr in [('FN', fn), ('TN', tn)]:
        if len(arr) > 1:
            kde = gaussian_kde(arr)
            ax2.plot(x_range, kde(x_range), color=colors[label],
                     label=f'{label} (n={len(arr)})', linewidth=2)
            ax2.fill_between(x_range, kde(x_range), alpha=0.15, color=colors[label])
    ax2.set_xlabel('Soft top-k score  [0, 1]')
    ax2.set_ylabel('Density')
    ax2.set_title('Rejected pairs (hard=0)\nFN = missed match   TN = correct rejection')
    ax2.set_xlim(0.0, 1.0)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Row 1 left: GT match (TP+FN) vs FP — full matrix ────────────────────
    gt_match = np.concatenate([arr for arr in [tp, fn] if len(arr) > 1])  # all gt=1
    groups_bottom = [('GT match  (gt=1)', gt_match, colors['TP']),
                     ('FP  (hard=1, gt=0)', fp,       colors['FP'])]

    def _draw_bottom(ax, ylim=None):
        for label, arr, color in groups_bottom:
            if len(arr) > 1:
                kde = gaussian_kde(arr)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2,
                        label=f'{label} (n={len(arr)})')
                ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)
        ax.set_xlabel('Soft top-k score  [0, 1]')
        ax.set_ylabel('Density')
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(0.0, ylim)

    ax3 = fig.add_subplot(gs[1, :])
    _draw_bottom(ax3)
    ax3.set_title('Full matrix: GT match vs FP\n(GT match = TP + FN)')

    plt.show(block=True)

    # Summary statistics
    print("\n--- Soft top-k score summary ---")
    for label, arr in [('TP', tp), ('FP', fp), ('FN', fn), ('TN', tn)]:
        if len(arr):
            print(f"  {label} (n={len(arr):5d}):  "
                  f"min={arr.min():.3f}  "
                  f"p10={np.percentile(arr,10):.3f}  "
                  f"median={np.median(arr):.3f}  "
                  f"p90={np.percentile(arr,90):.3f}  "
                  f"max={arr.max():.3f}")



def _save_single_heatmap(affinity, sinkhorn, soft, hard, gt, idx, save_dir):
    """Save a 4-panel heatmap for one pair with cell values annotated."""
    # N1, N2 = soft.shape
    # gt_rows, gt_cols = np.where(gt == 1)

    # cell_size = max(1.0, min(2.0, 12 / max(N1, N2)))
    # fig_w = min(4 * N2 * cell_size + 4, 36)
    # fig_h = min(N1 * cell_size + 2, 12)
    # fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h))
    # fig.suptitle(f'Pair {idx} — N1={N1}, N2={N2}, GT matches={len(gt_rows)}', fontsize=12)

    # matrices = [
    #     (affinity, 'Affinity\n(sim_normed)',              'RdYlGn'),
    #     (sinkhorn, 'Doubly stochastic\n(first Sinkhorn)', 'Blues'),
    #     (soft,     'Soft top-k scores',                   'Blues'),
    #     (hard,     'Hard permutation\n(final selection)',  'Greys'),
    # ]

    # for ax, (mat, title, cmap) in zip(axes, matrices):
    #     im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=mat.min(), vmax=mat.max())
    #     thresh = (mat.max() + mat.min()) / 2
    #     fmt = '.2f' if mat.dtype.kind == 'f' else 'd'
    #     for i in range(N1):
    #         for j in range(N2):
    #             val = mat[i, j]
    #             color = 'white' if val < thresh else 'black'
    #             ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
    #                     fontsize=7, color=color, fontweight='bold')
    #     ax.scatter(gt_cols, gt_rows, c='none', s=200, marker='o',
    #                edgecolors='red', linewidths=2, zorder=5, label='GT match')
    #     ax.set_title(title, fontsize=10)
    #     ax.set_xlabel('G2 nodes (j)')
    #     ax.set_ylabel('G1 nodes (i)')
    #     ax.set_xticks(range(N2))
    #     ax.set_yticks(range(N1))
    #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # axes[0].legend(fontsize=8, loc='upper right')
    # plt.tight_layout()
    # save_path = os.path.join(save_dir, f'matrices_pair{idx:03d}.png')
    # plt.savefig(save_path, dpi=100, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")
    pass


def plot_intermediate_distributions(aff_gt, aff_non_gt, sk_gt, sk_non_gt, hung_gt, hung_non_gt):
    """
    Plot score distributions of the affinity matrix, the doubly-stochastic Sinkhorn
    matrix, and the hard permutation matrix from Hungarian — all split by GT label
    (GT match in green, GT non-match in red).
    """
    from scipy.stats import gaussian_kde

    colors = {'gt': '#2ca02c', 'non_gt': '#d62728'}

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('Intermediate matrix score distributions (by GT label)', fontsize=13)
    gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.3)

    def _draw_kde(ax, gt_arr, non_gt_arr, title, xlabel):
        for label, arr, color in [
            ('GT match (gt=1)',      gt_arr,     colors['gt']),
            ('GT non-match (gt=0)', non_gt_arr, colors['non_gt']),
        ]:
            if len(arr) > 1:
                kde = gaussian_kde(arr)
                x_range = np.linspace(arr.min(), arr.max(), 500)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2,
                        label=f'{label} (n={len(arr)})')
                ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_discrete(ax, gt_arr, non_gt_arr, title, xlabel):
        """Impulse (stem) plot for discrete {0, 1} distributions."""
        x = np.array([0, 1])
        for label, arr, color in zip(
            ['GT match (gt=1)', 'GT non-match (gt=0)'],
            [gt_arr, non_gt_arr],
            [colors['gt'], colors['non_gt']],
        ):
            if len(arr) > 0:
                counts = np.array([(arr == v).sum() for v in [0, 1]], dtype=float)
                proportions = counts / counts.sum() if counts.sum() > 0 else counts
                markerline, stemlines, baseline = ax.stem(
                    x, proportions, linefmt=color, markerfmt='o',
                    basefmt=' ', label=f'{label} (n={len(arr)})'
                )
                markerline.set_color(color)
                markerline.set_markersize(8)
                stemlines.set_linewidth(2.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0 (not assigned)', '1 (assigned)'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Proportion')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_kde(ax1, aff_gt, aff_non_gt,
              title='Normalised affinity matrix\n(before first Sinkhorn)',
              xlabel='Affinity score (instance-normalised)')

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_kde(ax2, sk_gt, sk_non_gt,
              title='Doubly stochastic matrix\n(after Sinkhorn, before Hungarian)',
              xlabel='Sinkhorn score  [0, 1]')

    ax3 = fig.add_subplot(gs[0, 2])
    _draw_discrete(ax3, hung_gt, hung_non_gt,
                   title='Hard permutation matrix\n(Hungarian output)',
                   xlabel='Assignment value  {0, 1}')

    plt.show(block=False)


def plot_hungarian_assigned_scores(sk_assigned_tp, sk_assigned_fp):
    """
    Plot the distribution of doubly-stochastic S scores at the positions
    that Hungarian selected (hard=1), split by GT label:
      - TP (hard=1, gt=1): correct assignments — want these high
      - FP (hard=1, gt=0): wrong assignments  — want these low

    This shows whether the model assigns high S scores to correct matches
    and low scores to wrong ones, directly indicating if a sinkhorn_threshold
    could separate TP from FP.
    """
    from scipy.stats import gaussian_kde

    colors = {'TP': '#2ca02c', 'FP': '#d62728'}

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('S scores at Hungarian-assigned positions (hard=1)', fontsize=13)

    for label, arr, color in [
        ('TP — correct assignments (gt=1)', sk_assigned_tp, colors['TP']),
        ('FP — wrong assignments  (gt=0)',  sk_assigned_fp, colors['FP']),
    ]:
        if len(arr) > 1:
            kde = gaussian_kde(arr)
            x_range = np.linspace(0.0, 1.0, 500)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2,
                    label=f'{label} (n={len(arr)})')
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)

    ax.set_xlabel('Sinkhorn score S[i,j]  [0, 1]')
    ax.set_ylabel('Density')
    #ax.set_title(')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def run_score_distribution_analysis(sinkhorn_threshold=None, max_pairs=None):
    """Load the PGM model + MSD dataset and plot soft top-k score distributions."""
    import sys
    import pickle

    GNN_PATH   = '/root/workspace/src/graph_matching_gnn/GNN'
    PGM_PATH   = '/root/workspace/src/graph_matching_gnn/graph_matching'
    MSD_PATH   = os.path.join(GNN_PATH, 'preprocessed', 'partial_graph_matching',
                              'ws_room_dropout_noise', 'test_dataset.pkl')

    for p in (PGM_PATH, GNN_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from PGM_class import PartialGraphMatching, MatchingModel_GATv2SinkhornTopK  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pgm = PartialGraphMatching(
        model_class=MatchingModel_GATv2SinkhornTopK,
        data_paths={
            'equal':   os.path.join(GNN_PATH, 'preprocessed', 'graph_matching', 'equal'),
            'partial': os.path.join(GNN_PATH, 'preprocessed', 'partial_graph_matching',
                                    'ws_room_dropout_noise'),
        },
        model_save_path=os.path.join(GNN_PATH, 'models', 'partial_graph_matching',
                                     'ws_room_dropout_noise'),
        device=device, in_dim=7,
    )
    pgm.load_best_model()
    print("PGM model loaded.")

    with open(MSD_PATH, 'rb') as f:
        msd_test_list = pickle.load(f)
    n = len(msd_test_list) if max_pairs is None else min(max_pairs, len(msd_test_list))
    print(f"MSD test set: {len(msd_test_list)} pairs — analysing {n}")

    matrices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_heatmaps')
    tp, fp, fn, tn, aff_gt, aff_non_gt, sk_gt, sk_non_gt, hung_gt, hung_non_gt, sk_assigned_tp, sk_assigned_fp = collect_soft_scores_from_msd(
        pgm.model, msd_test_list,
        sinkhorn_threshold=sinkhorn_threshold,
        max_pairs=max_pairs,
        save_heatmaps_dir=matrices_dir,
        heatmap_pairs=min(3, n),
    )

    plot_intermediate_distributions(aff_gt, aff_non_gt, sk_gt, sk_non_gt, hung_gt, hung_non_gt)
    plot_hungarian_assigned_scores(sk_assigned_tp, sk_assigned_fp)
    plot_soft_score_distribution(tp, fp, fn, tn)


def main():
    parser = argparse.ArgumentParser(description='Plot graph matching results from JSON file')
    parser.add_argument('json_file', nargs='?', default='latest_results.json',
                        help='JSON file (default: latest_results.json)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display plots (only save)')
    parser.add_argument('--no-scatter', action='store_true',
                        help='Show surfaces only, hide scatter points')
    parser.add_argument('--score-dist', action='store_true',
                        help='Plot soft top-k score distributions (runs model on MSD test set)')
    parser.add_argument('--sinkhorn-threshold', type=float, default=None,
                        help='Post-Sinkhorn threshold applied to S before Hungarian during score-dist analysis')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Limit number of MSD pairs analysed in --score-dist mode')
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir

    if args.score_dist:
        run_score_distribution_analysis(
            sinkhorn_threshold=args.sinkhorn_threshold,
            max_pairs=args.max_pairs,
        )

    json_filepath = (os.path.join(results_dir, args.json_file)
                     if not os.path.dirname(args.json_file)
                     else args.json_file)

    print(f"Loading results from: {json_filepath}")
    data = load_results(json_filepath)
    if data is None:
        sys.exit(1)
    print(f"Loaded {len(data['experiments'])} experiments")

    (metrics_by_rooms, success_data, timing_data,
     solution_count_data, metadata, raw_points) = process_data(data)
    room_counts, metric_names, metric_labels, stats = calculate_statistics(
        metrics_by_rooms, success_data, timing_data, solution_count_data)

    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path    = os.path.join(results_dir, f"graph_matching_3d_{timestamp}.png")
    summary_path = os.path.join(results_dir, f"graph_matching_summary_{timestamp}.txt")

    save_summary_statistics(room_counts, metric_names, metric_labels,
                            stats, metrics_by_rooms, metadata, summary_path)
    print_numerical_summary(room_counts, metric_names, metric_labels,
                            stats, metrics_by_rooms)

    current_dataset  = metadata.get('dataset')
    threshold_points = collect_threshold_points(results_dir, current_dataset=current_dataset)

    fig = create_3d_plots(raw_points, metadata, plot_path,
                          show_scatter=not args.no_scatter,
                          threshold_points=threshold_points)

    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    print(f"\nDone.")
    print(f"  3D plot : {plot_path}")
    print(f"  Summary : {summary_path}")


if __name__ == "__main__":
    main()
