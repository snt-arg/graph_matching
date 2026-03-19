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


def main():
    parser = argparse.ArgumentParser(description='Plot graph matching results from JSON file')
    parser.add_argument('json_file', nargs='?', default='latest_results.json',
                        help='JSON file (default: latest_results.json)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display plots (only save)')
    parser.add_argument('--no-scatter', action='store_true',
                        help='Show surfaces only, hide scatter points')
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_dir  = script_dir

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
