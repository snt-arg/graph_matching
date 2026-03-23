#!/usr/bin/env python3
"""
Graph Matching Results Plotter

This script reads JSON results from graph matching experiments and creates 
comprehensive visualizations of performance metrics.

Usage:
    python plot_results.py [results_file.json]
    python plot_results.py --json-file results_file.json

If no file is specified, the script prompts you to select one from the results directory
with arrow keys (or numeric fallback if the terminal does not support it).
"""

import json
import os
import sys
import argparse
import curses
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from collections import defaultdict


def load_results(json_filepath):
    """Load experiment results from JSON file."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {json_filepath} not found!")
        return None


def select_results_file(results_dir, requested_file=None):
    """Resolve JSON results file path, optionally prompting the user to choose."""
    if requested_file:
        if not os.path.dirname(requested_file):
            return os.path.join(results_dir, requested_file)
        return requested_file

    json_files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith('.json') and os.path.isfile(os.path.join(results_dir, f))]
    )

    if not json_files:
        return os.path.join(results_dir, 'latest_results.json')

    default_idx = 0
    for idx, name in enumerate(json_files, start=1):
        if name == 'latest_results.json':
            default_idx = idx - 1

    def _arrow_menu(files, selected_idx):
        """Arrow-key picker UI using curses. Returns selected index."""

        def _run(stdscr):
            current = selected_idx
            curses.curs_set(0)
            stdscr.keypad(True)

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = "Select a results JSON file (Up/Down, Enter)"
                hint = "Press q to use default (latest_results.json if available)."

                stdscr.addnstr(0, 0, title, max(w - 1, 1), curses.A_BOLD)
                stdscr.addnstr(1, 0, hint, max(w - 1, 1), curses.A_DIM)

                max_visible = max(h - 3, 1)
                start = max(0, min(current - max_visible // 2, len(files) - max_visible))
                end = min(len(files), start + max_visible)

                for row, i in enumerate(range(start, end), start=3):
                    prefix = "> " if i == current else "  "
                    line = f"{prefix}{files[i]}"
                    attr = curses.A_REVERSE if i == current else curses.A_NORMAL
                    stdscr.addnstr(row, 0, line, max(w - 1, 1), attr)

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord('k')):
                    current = (current - 1) % len(files)
                elif key in (curses.KEY_DOWN, ord('j')):
                    current = (current + 1) % len(files)
                elif key in (10, 13, curses.KEY_ENTER):
                    return current
                elif key in (ord('q'), 27):
                    return selected_idx

        return curses.wrapper(_run)

    selected_idx = default_idx
    used_arrow_menu = False
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            selected_idx = _arrow_menu(json_files, default_idx)
            used_arrow_menu = True
        except Exception:
            used_arrow_menu = False

    if not used_arrow_menu:
        print("Select a results JSON file:")
        for idx, name in enumerate(json_files, start=1):
            print(f"  {idx}. {name}")
        default_choice = default_idx + 1
        prompt = f"Enter number [default {default_choice}]: "
        choice = input(prompt).strip()
        if not choice:
            selected_idx = default_idx
        else:
            try:
                selected_idx = int(choice) - 1
                if selected_idx < 0 or selected_idx >= len(json_files):
                    raise ValueError
            except ValueError:
                print(f"Invalid choice '{choice}'. Using default: {json_files[default_idx]}")
                selected_idx = default_idx

    selected = json_files[selected_idx]

    return os.path.join(results_dir, selected)


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
        ('num_solutions', '#Solutions'),
    ]

    fig = plt.figure(figsize=(28, 20))
    fig.patch.set_facecolor('white')
    fig.suptitle(fig_title, fontsize=16)
    experiment_types = sorted(raw_points_by_type.keys())

    for plot_idx, (field, zlabel) in enumerate(plot_specs):
        ax = fig.add_subplot(3, 3, plot_idx + 1, projection='3d')
        ax.patch.set_facecolor('white')
        
        # Set 3D axes panes to white with light grey grid lines
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        
        # Set grid line colors to light grey for visibility
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        field_x_values = []
        field_y_values = []

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

            field_x_values.extend(xs_v.tolist())
            field_y_values.extend(ys_v.tolist())

            surface_drawn = False

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
                    ax.plot_surface(Xi, Yi, Zi, alpha=0.35, color=color,
                                    linewidth=0, antialiased=True)
                    surface_drawn = True

                    # For num_solutions field, plot intersection at z=1.2
                    if field == 'num_solutions':
                        try:
                            contours = ax.contour(Xi, Yi, Zi, levels=[1.2], colors=[color],
                                                 linewidths=2, alpha=0.8)
                            for collection in contours.collections:
                                for path in collection.get_paths():
                                    vertices = path.vertices
                                    if len(vertices) > 1:
                                        ax.plot(vertices[:, 0], vertices[:, 1],
                                               [1.2] * len(vertices),
                                               color=color, linewidth=2.5, alpha=0.8)
                        except Exception:
                            pass
                except Exception:
                    pass

            if show_scatter:
                ax.scatter(xs_v, ys_v, zs_v, color=color, s=20,
                           label=label, depthshade=True)
            elif not surface_drawn:
                # Surface interpolation can fail for sparse/degenerate point sets.
                # In no-scatter mode, draw a minimal fallback so plots are not blank.
                ax.scatter(xs_v, ys_v, zs_v, color=color, s=10,
                           label=label, depthshade=True, alpha=0.8)
            elif plot_idx == 0:
                # Still need a handle for the legend when scatter is hidden
                ax.scatter([], [], [], color=color, s=20, label=label)

        if field == 'num_solutions':
            # Keep the solutions axis grounded at zero for easier comparison.
            _, current_z_max = ax.get_zlim()
            ax.set_zlim(bottom=0.0, top=max(current_z_max, 1.5))

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_zlabel(zlabel, fontsize=8)
        ax.set_title(zlabel, fontsize=10, fontweight='bold')
        
        # For num_solutions plot, use integer-only z-axis ticks
        if field == 'num_solutions':
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        
        if plot_idx == 0:
            ax.legend(fontsize=7)

    return fig


def create_3d_plots(raw_points_by_type, metadata, save_path=None, show_scatter=True):
    """3D surface plots: X = n_rooms_s, Y = n_rooms_a, Z = metric."""
    fig = _make_3d_subplots(
        raw_points_by_type,
        fig_title='Graph Matching: 3D View  (#S-Rooms \u00d7 #A-Rooms)',
        xlabel='#S-Rooms',
        ylabel='#A-Rooms',
        x_getter=lambda p: p['n_rooms_s'],
        y_getter=lambda p: p['n_rooms_a'],
        save_path=None,
        show_scatter=show_scatter,
    )

    # Summary info in the 9th cell
    ax_info = fig.add_subplot(3, 3, 9)
    ax_info.patch.set_facecolor('white')
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
    ax_info.patch.set_facecolor('white')
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
                       help='JSON file containing results (legacy positional argument)')
    parser.add_argument('--json-file', dest='json_file_opt', default=None,
                       help='JSON file containing results; if omitted, select interactively with arrows')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots (generate in memory only)')
    parser.add_argument('--no-scatter', action='store_true',
                       help='Hide individual data-point spheres; show surfaces only')

    args = parser.parse_args()
    
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    
    requested_file = args.json_file_opt
    if requested_file is None and args.json_file != 'latest_results.json':
        # Preserve compatibility for explicit positional usage.
        requested_file = args.json_file

    json_filepath = select_results_file(results_dir, requested_file)
    
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
    
    show_scatter = not args.no_scatter

    # Figure 1 – absolute room counts (X=S-rooms, Y=A-rooms, Z=metric)
    fig1 = create_3d_plots(raw_points, metadata, save_path=None, show_scatter=show_scatter)

    # Figure 2 – normalised axes (X=S/A ratio, Y=%obj nodes, Z=metric)
    fig2 = create_3d_normalized_plots(raw_points, metadata, save_path=None, show_scatter=show_scatter)
    
    # Print numerical summary
    print_numerical_summary(room_counts, metric_names, metric_labels,
                           stats_by_type, experiment_types, metrics_by_rooms_and_type)
    
    # Display plots unless disabled
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        print("Plots generated but not displayed (--no-display flag used)")
    
    print(f"\nResults processed successfully!")
    print("  - No .png/.txt files were saved")


if __name__ == "__main__":
    main()