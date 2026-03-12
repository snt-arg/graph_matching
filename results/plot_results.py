#!/usr/bin/env python3
"""
Graph Matching Results Plotter

This script reads JSON results from graph matching experiments and creates 
comprehensive visualizations of performance metrics.

Usage:
    python plot_results.py [results_file.json]
    
If no file is specified, it will look for 'latest_results.json'
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
    """Process raw data and group by number of rooms and experiment type."""
    experiments = data['experiments']
    metadata = data['metadata']
    
    # Group metrics by number of rooms and experiment type
    metrics_by_rooms_and_type = defaultdict(lambda: defaultdict(list))
    # Also track additional metrics
    success_data = defaultdict(lambda: defaultdict(list))
    timing_data = defaultdict(lambda: defaultdict(list))
    solution_count_data = defaultdict(lambda: defaultdict(list))

    # Raw per-experiment points for 3D scatter/surface plots
    raw_points = defaultdict(list)
    _cls_metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']

    for experiment in experiments:
        n_rooms_s = experiment['n_rooms_s_graphs']
        n_rooms_a = experiment.get('n_rooms_a_graphs', n_rooms_s)
        pct_obj = experiment.get('pct_object_nodes', 0.0)
        exp_type = experiment.get('experiment_type', 'unknown')
        
        # Extract matching time
        matching_time = experiment.get('matching_time', 0.0)
        timing_data[exp_type][n_rooms_s].append(matching_time)
        
        # Check success: use explicit 'success' field if present, else infer from metrics
        if 'success' in experiment:
            is_success = experiment['success']
        else:
            is_success = 'metrics' in experiment and bool(experiment['metrics'])
        success_data[exp_type][n_rooms_s].append(1 if is_success else 0)
        
        has_metrics = ('metrics' in experiment and 
                       bool(experiment['metrics']) and 
                       'precision' in experiment['metrics'])
        if has_metrics:
            metrics = experiment['metrics']
            metrics_by_rooms_and_type[exp_type][n_rooms_s].append(metrics)

        # num_solutions is always at the top level of the experiment, not inside metrics
        num_solutions = experiment.get('num_solutions', 0)
        solution_count_data[exp_type][n_rooms_s].append(num_solutions)

        # Collect raw point for 3D plots
        point = {
            'n_rooms_s': n_rooms_s,
            'n_rooms_a': n_rooms_a,
            'pct_obj': pct_obj,
            'success': 1 if is_success else 0,
            'time': matching_time,
            'num_solutions': num_solutions,
        }
        for m in _cls_metrics:
            point[m] = (experiment['metrics'][m]
                        if has_metrics and m in experiment['metrics']
                        else np.nan)
        raw_points[exp_type].append(point)
    
    return metrics_by_rooms_and_type, success_data, timing_data, solution_count_data, metadata, raw_points


def calculate_statistics(metrics_by_rooms_and_type, success_data, timing_data, solution_count_data):
    """Calculate statistics for all metrics including performance, success rates, timing, and solution counts."""
    
    # Get all experiment types from ALL data sources, not just those with classification metrics
    all_exp_types = set()
    for d in [metrics_by_rooms_and_type, success_data, timing_data, solution_count_data]:
        all_exp_types.update(d.keys())
    experiment_types = sorted(all_exp_types)

    all_room_counts = set()
    for exp_type in experiment_types:
        for d in [metrics_by_rooms_and_type, success_data, timing_data, solution_count_data]:
            all_room_counts.update(d[exp_type].keys())
    room_counts = sorted(all_room_counts)
    
    # Performance metrics to analyze
    metric_names = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
    
    # Calculate statistics for each experiment type
    stats_by_type = {}
    for exp_type in experiment_types:
        # Performance metrics
        averaged_metrics = {metric: [] for metric in metric_names}
        std_metrics = {metric: [] for metric in metric_names}
        
        # Success rates
        success_rates = []
        
        # Timing statistics
        avg_times = []
        std_times = []
        
        # Solution counts
        avg_solutions = []
        std_solutions = []
        
        for room_count in room_counts:
            # Performance metrics
            if room_count in metrics_by_rooms_and_type[exp_type]:
                metrics_list = metrics_by_rooms_and_type[exp_type][room_count]
                for metric in metric_names:
                    values = [m[metric] for m in metrics_list]
                    averaged_metrics[metric].append(np.mean(values))
                    std_metrics[metric].append(np.std(values))
            else:
                for metric in metric_names:
                    averaged_metrics[metric].append(np.nan)
                    std_metrics[metric].append(np.nan)
            
            # Success rates
            if room_count in success_data[exp_type]:
                successes = success_data[exp_type][room_count]
                success_rate = np.mean(successes) * 100  # Convert to percentage
                success_rates.append(success_rate)
            else:
                success_rates.append(np.nan)
            
            # Timing statistics
            if room_count in timing_data[exp_type]:
                times = timing_data[exp_type][room_count]
                avg_times.append(np.mean(times))
                std_times.append(np.std(times))
            else:
                avg_times.append(np.nan)
                std_times.append(np.nan)
            
            # Solution counts
            if room_count in solution_count_data[exp_type]:
                solutions = solution_count_data[exp_type][room_count]
                avg_solutions.append(np.mean(solutions))
                std_solutions.append(np.std(solutions))
            else:
                avg_solutions.append(np.nan)
                std_solutions.append(np.nan)
        
        stats_by_type[exp_type] = {
            'averaged_metrics': averaged_metrics,
            'std_metrics': std_metrics,
            'success_rates': success_rates,
            'avg_times': avg_times,
            'std_times': std_times,
            'avg_solutions': avg_solutions,
            'std_solutions': std_solutions
        }
    
    return room_counts, metric_names, metric_labels, stats_by_type, experiment_types


def _attach_expand_on_dblclick(fig):
    """Double-click a subplot to expand it full-window; double-click again to restore."""
    axes_list = fig.get_axes()
    orig_pos = {ax: ax.get_position() for ax in axes_list}
    orig_vis = {ax: ax.get_visible() for ax in axes_list}
    state = [None]  # state[0] = currently expanded ax, or None

    def on_dblclick(event):
        if not event.dblclick:
            return
        dpi = fig.dpi
        fx = event.x / (fig.get_figwidth() * dpi)
        fy = event.y / (fig.get_figheight() * dpi)

        if state[0] is not None:
            # Restore all axes
            for ax in axes_list:
                ax.set_position(orig_pos[ax])
                ax.set_visible(orig_vis[ax])
            state[0] = None
        else:
            # Find the axes that was clicked
            clicked = None
            for ax in axes_list:
                pos = orig_pos[ax]
                if pos.x0 <= fx <= pos.x1 and pos.y0 <= fy <= pos.y1:
                    clicked = ax
                    break
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


def _make_3d_subplots(raw_points_by_type, fig_title, xlabel, ylabel,
                      x_getter, y_getter, save_path, show_scatter=True):
    """Shared helper: create a 3x3 figure of 3D surface+scatter subplots.

    x_getter / y_getter are callables that accept a raw point dict and return
    the X or Y coordinate for that point.
    show_scatter: if False, only the interpolated surface is drawn.
    """
    type_colors = {
        'with_objects': '#1f77b4',
        'no_objects': '#ff7f0e',
        'unknown': '#2ca02c',
    }
    type_labels_map = {
        'with_objects': 'With Objects',
        'no_objects': 'Without Objects',
        'unknown': 'Unknown',
    }
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
    NUM_SOLUTIONS_THRESHOLD = 1.5  # highlight points at or below this value

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(fig_title, fontsize=16)
    experiment_types = sorted(raw_points_by_type.keys())

    for plot_idx, (field, zlabel) in enumerate(plot_specs):
        ax = fig.add_subplot(3, 3, plot_idx + 1, projection='3d')

        for exp_type in experiment_types:
            points = raw_points_by_type[exp_type]
            color = type_colors.get(exp_type, '#2ca02c')
            label = type_labels_map.get(exp_type, exp_type)

            xs = np.array([x_getter(p) for p in points], dtype=float)
            ys = np.array([y_getter(p) for p in points], dtype=float)
            zs = np.array([p[field] for p in points], dtype=float)

            valid = ~np.isnan(zs)
            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
            if len(xs_v) == 0:
                continue

            # Attempt interpolated surface when data spans 2D
            unique_xy = np.unique(np.column_stack([xs_v, ys_v]), axis=0)
            if (len(unique_xy) >= 6
                    and len(np.unique(xs_v)) >= 2
                    and len(np.unique(ys_v)) >= 2):
                xi = np.linspace(xs_v.min(), xs_v.max(), 30)
                yi = np.linspace(ys_v.min(), ys_v.max(), 30)
                Xi, Yi = np.meshgrid(xi, yi)
                try:
                    # Average z values for duplicate (x, y) pairs so the surface
                    # reflects the mean of all samples at each grid position.
                    xy_to_zs = defaultdict(list)
                    for x, y, z in zip(xs_v, ys_v, zs_v):
                        xy_to_zs[(x, y)].append(z)
                    xs_agg = np.array([k[0] for k in xy_to_zs])
                    ys_agg = np.array([k[1] for k in xy_to_zs])
                    zs_agg = np.array([np.mean(v) for v in xy_to_zs.values()])
                    Zi = griddata((xs_agg, ys_agg), zs_agg, (Xi, Yi), method='linear')
                    if field == 'num_solutions':
                        # Use a two-tone colormap: red below threshold, normal color above
                        import matplotlib.colors as mcolors
                        cmap_colors = ['#d62728', '#d62728', color, color]
                        cmap_nodes = [0.0, NUM_SOLUTIONS_THRESHOLD, NUM_SOLUTIONS_THRESHOLD, max(zs_agg.max(), NUM_SOLUTIONS_THRESHOLD + 0.01)]
                        norm_max = cmap_nodes[-1]
                        norm_nodes = [v / norm_max for v in cmap_nodes]
                        cmap = mcolors.LinearSegmentedColormap.from_list(
                            'thresh', list(zip(norm_nodes, cmap_colors)))
                        z_norm = mcolors.Normalize(vmin=0, vmax=norm_max)
                        fcolors = cmap(z_norm(Zi))
                        ax.plot_surface(Xi, Yi, Zi, facecolors=fcolors, alpha=0.5,
                                        linewidth=0, antialiased=True)
                    else:
                        ax.plot_surface(Xi, Yi, Zi, alpha=0.35, color=color,
                                        linewidth=0, antialiased=True)
                except Exception:
                    pass

            if show_scatter:
                ax.scatter(xs_v, ys_v, zs_v, color=color, s=20,
                           label=label, depthshade=True)
            elif plot_idx == 0:
                # Still need a handle for the legend when scatter is hidden
                ax.scatter([], [], [], color=color, s=20, label=label)

        ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
        ax.set_zlabel(zlabel, fontsize=8, labelpad=3)
        ax.set_title(zlabel, fontsize=10, fontweight='bold')
        if plot_idx == 0:
            ax.legend(fontsize=7)

    return fig


def create_3d_plots(raw_points_by_type, metadata, save_path=None, show_scatter=True):
    """3D surface plots: X = n_rooms_s, Y = n_rooms_a, Z = metric."""
    fig = _make_3d_subplots(
        raw_points_by_type,
        fig_title='Graph Matching: 3D View  (S-Rooms \u00d7 A-Rooms)',
        xlabel='S-Rooms',
        ylabel='A-Rooms',
        x_getter=lambda p: p['n_rooms_s'],
        y_getter=lambda p: p['n_rooms_a'],
        save_path=None,
        show_scatter=show_scatter,
    )

    # Summary info in the 9th cell
    ax_info = fig.add_subplot(3, 3, 9)
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
        print(f"3D plot (abs rooms) saved to: {save_path}")

    return fig


def create_3d_normalized_plots(raw_points_by_type, metadata, save_path=None, show_scatter=True):
    """3D surface plots: X = n_rooms_s/n_rooms_a ratio, Y = pct_object_nodes, Z = metric.

    All experiment types are merged into one series so that no_objects (pct_obj=0)
    and with_objects (pct_obj>0) appear as different regions of the same surface.
    """
    merged = {'all': [p for pts in raw_points_by_type.values() for p in pts]}
    fig = _make_3d_subplots(
        merged,
        fig_title='Graph Matching: 3D View  (S/A Ratio \u00d7 % Object Nodes)',
        xlabel='S/A Ratio',
        ylabel='% Obj Nodes',
        x_getter=lambda p: p['n_rooms_s'] / max(p['n_rooms_a'], 1),
        y_getter=lambda p: p['pct_obj'],
        save_path=None,
        show_scatter=show_scatter,
    )

    ax_info = fig.add_subplot(3, 3, 9)
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
        print(f"3D plot (normalized) saved to: {save_path}")

    return fig


def print_numerical_summary(room_counts, metric_names, metric_labels, 
                           stats_by_type, experiment_types, metrics_by_rooms_and_type):
    """Print detailed numerical summary of results by experiment type."""
    print("\n" + "="*80)
    print("SUMMARY: AVERAGED METRICS BY NUMBER OF ROOMS AND EXPERIMENT TYPE")
    print("="*80)
    
    type_labels = {'with_objects': 'With Objects', 'no_objects': 'Without Objects', 'unknown': 'Unknown'}
    
    for exp_type in experiment_types:
        print(f"\n{type_labels.get(exp_type, exp_type.title())} Experiments:")
        print("-" * 50)
        
        if exp_type not in stats_by_type:
            print("  No data available")
            continue
        
        stats = stats_by_type[exp_type]
        averaged_metrics = stats['averaged_metrics']
        std_metrics = stats['std_metrics']
        
        for idx, room_count in enumerate(room_counts):
            # Always print room entry if we have any data for it
            success_rate = stats['success_rates'][idx]
            avg_time = stats['avg_times'][idx]
            avg_solutions = stats['avg_solutions'][idx]
            has_any_data = not (np.isnan(success_rate) and np.isnan(avg_time))

            if not has_any_data:
                continue

            has_classification = room_count in metrics_by_rooms_and_type[exp_type]
            n_total = int(round(len(metrics_by_rooms_and_type[exp_type].get(room_count, [])) or
                                (stats['success_rates'][idx] / 100 if not np.isnan(success_rate) else 0)))
            n_class = len(metrics_by_rooms_and_type[exp_type].get(room_count, []))
            print(f"\n  Rooms: {room_count}")

            if not np.isnan(success_rate):
                print(f"    {'Success Rate':12s}: {success_rate:.1f}%")
            if not np.isnan(avg_time):
                print(f"    {'Avg Time':12s}: {avg_time:.4f}s ± {stats['std_times'][idx]:.4f}s")
            if not np.isnan(avg_solutions):
                print(f"    {'Avg Solutions':13s}: {avg_solutions:.2f} ± {stats['std_solutions'][idx]:.2f}")

            if has_classification:
                print(f"    --- Classification metrics (N={n_class} unambiguous) ---")
                for metric, label in zip(metric_names, metric_labels):
                    mean_val = averaged_metrics[metric][idx]
                    std_val = std_metrics[metric][idx]
                    if not np.isnan(mean_val):
                        print(f"    {label:12s}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"    Classification metrics: N/A (no unambiguous solution)")


def save_summary_statistics(room_counts, metric_names, metric_labels, 
                           stats_by_type, experiment_types, metrics_by_rooms_and_type,
                           metadata, save_path):
    """Save detailed statistics to a text file."""
    type_labels = {'with_objects': 'With Objects', 'no_objects': 'Without Objects', 'unknown': 'Unknown'}
    
    with open(save_path, 'w') as f:
        f.write("GRAPH MATCHING PERFORMANCE ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Generated: {metadata['timestamp']}\n")
        f.write(f"Dataset: {metadata['dataset_file']}\n")
        f.write(f"Total Experiments: {metadata['total_experiments']}\n\n")
        
        f.write("AVERAGED METRICS BY NUMBER OF ROOMS AND EXPERIMENT TYPE\n")
        f.write("-" * 60 + "\n\n")
        
        for exp_type in experiment_types:
            f.write(f"{type_labels.get(exp_type, exp_type.title())} Experiments:\n")
            f.write("-" * 40 + "\n")
            
            if exp_type not in stats_by_type:
                f.write("No data available\n\n")
                continue
                
            averaged_metrics = stats_by_type[exp_type]['averaged_metrics']
            std_metrics = stats_by_type[exp_type]['std_metrics']
            
            for idx, room_count in enumerate(room_counts):
                success_rate = stats_by_type[exp_type]['success_rates'][idx]
                avg_time = stats_by_type[exp_type]['avg_times'][idx]
                avg_solutions = stats_by_type[exp_type]['avg_solutions'][idx]
                has_any_data = not (np.isnan(success_rate) and np.isnan(avg_time))
                if not has_any_data:
                    continue

                f.write(f"\nRooms: {room_count}\n")
                if not np.isnan(success_rate):
                    f.write(f"  {'Success Rate':12s}: {success_rate:.1f}%\n")
                if not np.isnan(avg_time):
                    f.write(f"  {'Avg Time':12s}: {avg_time:.4f}s ± {stats_by_type[exp_type]['std_times'][idx]:.4f}s\n")
                if not np.isnan(avg_solutions):
                    f.write(f"  {'Avg Solutions':13s}: {avg_solutions:.2f} ± {stats_by_type[exp_type]['std_solutions'][idx]:.2f}\n")

                if room_count in metrics_by_rooms_and_type[exp_type]:
                    n_class = len(metrics_by_rooms_and_type[exp_type][room_count])
                    f.write(f"  --- Classification metrics (N={n_class} unambiguous) ---\n")
                    for metric, label in zip(metric_names, metric_labels):
                        mean_val = averaged_metrics[metric][idx]
                        std_val = std_metrics[metric][idx]
                        if not np.isnan(mean_val):
                            f.write(f"  {label:12s}: {mean_val:.4f} ± {std_val:.4f}\n")
                else:
                    f.write(f"  Classification metrics: N/A (no unambiguous solution)\n")
            f.write("\n")
    
    print(f"Summary statistics saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot graph matching results from JSON file')
    parser.add_argument('json_file', nargs='?', default='latest_results.json',
                       help='JSON file containing results (default: latest_results.json)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots (only save them)')
    parser.add_argument('--no-scatter', action='store_true',
                       help='Hide individual data-point spheres; show surfaces only')

    args = parser.parse_args()
    
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    
    # If only filename provided, look in results directory
    if not os.path.dirname(args.json_file):
        json_filepath = os.path.join(results_dir, args.json_file)
    else:
        json_filepath = args.json_file
    
    # Load and process data
    print(f"Loading results from: {json_filepath}")
    data = load_results(json_filepath)
    
    if data is None:
        sys.exit(1)
    
    print(f"Loaded {len(data['experiments'])} experiments")
    
    # Process data
    (metrics_by_rooms_and_type, success_data, timing_data,
     solution_count_data, metadata, raw_points) = process_data(data)
    room_counts, metric_names, metric_labels, stats_by_type, experiment_types = calculate_statistics(
        metrics_by_rooms_and_type, success_data, timing_data, solution_count_data)
    
    # Create output filenames with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_3d_filename      = f"graph_matching_3d_{timestamp}.png"
    plot_norm_filename    = f"graph_matching_3d_normalized_{timestamp}.png"
    summary_filename      = f"graph_matching_summary_{timestamp}.txt"
    
    plot_3d_path   = os.path.join(results_dir, plot_3d_filename)
    plot_norm_path = os.path.join(results_dir, plot_norm_filename)
    summary_path   = os.path.join(results_dir, summary_filename)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    show_scatter = not args.no_scatter

    # Figure 1 – absolute room counts (X=S-rooms, Y=A-rooms, Z=metric)
    fig1 = create_3d_plots(raw_points, metadata, plot_3d_path, show_scatter=show_scatter)

    # Figure 2 – normalised axes (X=S/A ratio, Y=%obj nodes, Z=metric)
    fig2 = create_3d_normalized_plots(raw_points, metadata, plot_norm_path, show_scatter=show_scatter)
    
    # Save summary statistics
    save_summary_statistics(room_counts, metric_names, metric_labels,
                           stats_by_type, experiment_types, metrics_by_rooms_and_type,
                           metadata, summary_path)
    
    # Print numerical summary
    print_numerical_summary(room_counts, metric_names, metric_labels,
                           stats_by_type, experiment_types, metrics_by_rooms_and_type)
    
    # Display plots unless disabled
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        print("Plots saved but not displayed (--no-display flag used)")
    
    print(f"\nResults processed successfully!")
    print(f"  - 3D plot (abs rooms): {plot_3d_path}")
    print(f"  - 3D plot (normalized): {plot_norm_path}")
    print(f"  - Summary: {summary_path}")


if __name__ == "__main__":
    main()