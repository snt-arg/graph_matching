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
import re
import numpy as np

# Use non-interactive backend by default to avoid displaying windows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from collections import defaultdict


# Global typography scale for figure text.
# Requested: make text larger while keeping plots readable.
TEXT_SCALE = 3.0
MAX_FONT_SIZE = 72
VISUAL_SCALE = 3.0  # Scale for line widths, marker sizes, etc.
PLOT_SPECS = [
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1_score', 'F1-Score'),
    ('accuracy', 'Accuracy'),
    ('specificity', 'Specificity'),
    ('success', 'Success (0/1)'),
    ('time', 'Matching Time (s)'),
    ('num_solutions', '#Solutions'),
]


def _fs(size):
    """Scale and quantize font sizes consistently."""
    return min(MAX_FONT_SIZE, max(1, int(round(size * TEXT_SCALE))))


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


def select_results_files(results_dir, requested_files=None):
    """Resolve one-or-many JSON files, optionally prompting for multi-select."""
    if requested_files:
        resolved = []
        for requested_file in requested_files:
            if not os.path.dirname(requested_file):
                resolved.append(os.path.join(results_dir, requested_file))
            else:
                resolved.append(requested_file)
        # Preserve order while removing duplicates
        return list(dict.fromkeys(resolved))

    json_files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith('.json') and os.path.isfile(os.path.join(results_dir, f))]
    )

    if not json_files:
        return [os.path.join(results_dir, 'latest_results.json')]

    default_idx = 0
    for idx, name in enumerate(json_files):
        if name == 'latest_results.json':
            default_idx = idx
            break

    def _arrow_multi_menu(files, default_index):
        """Arrow-key multi picker using curses. Space toggles selection."""

        def _run(stdscr):
            current = default_index
            selected = {default_index}
            curses.curs_set(0)
            stdscr.keypad(True)

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = 'Select JSON files (Up/Down, Space toggle, Enter confirm)'
                hint = 'Press q to use default only (latest_results.json if available).'

                stdscr.addnstr(0, 0, title, max(w - 1, 1), curses.A_BOLD)
                stdscr.addnstr(1, 0, hint, max(w - 1, 1), curses.A_DIM)

                max_visible = max(h - 3, 1)
                start = max(0, min(current - max_visible // 2, len(files) - max_visible))
                end = min(len(files), start + max_visible)

                for row, i in enumerate(range(start, end), start=3):
                    mark = '[x]' if i in selected else '[ ]'
                    pointer = '>' if i == current else ' '
                    line = f"{pointer} {mark} {files[i]}"
                    attr = curses.A_REVERSE if i == current else curses.A_NORMAL
                    stdscr.addnstr(row, 0, line, max(w - 1, 1), attr)

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord('k')):
                    current = (current - 1) % len(files)
                elif key in (curses.KEY_DOWN, ord('j')):
                    current = (current + 1) % len(files)
                elif key == ord(' '):
                    if current in selected:
                        selected.remove(current)
                    else:
                        selected.add(current)
                elif key in (10, 13, curses.KEY_ENTER):
                    if not selected:
                        selected = {default_index}
                    return sorted(selected)
                elif key in (ord('q'), 27):
                    return [default_index]

        return curses.wrapper(_run)

    selected_indices = [default_idx]
    used_arrow_menu = False
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            selected_indices = _arrow_multi_menu(json_files, default_idx)
            used_arrow_menu = True
        except Exception:
            used_arrow_menu = False

    if not used_arrow_menu:
        print('Select JSON files (comma-separated numbers, e.g., 1,3,5):')
        for idx, name in enumerate(json_files, start=1):
            marker = '*' if (idx - 1) == default_idx else ' '
            print(f" {marker} {idx}. {name}")
        default_choice = str(default_idx + 1)
        choice = input(f"Enter numbers [default {default_choice}]: ").strip()
        if not choice:
            selected_indices = [default_idx]
        else:
            parsed = []
            for token in choice.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    i = int(token) - 1
                except ValueError:
                    continue
                if 0 <= i < len(json_files):
                    parsed.append(i)
            selected_indices = sorted(set(parsed)) if parsed else [default_idx]

    selected = [json_files[i] for i in selected_indices]
    return [os.path.join(results_dir, name) for name in selected]


def _expand_axis_range(values, pad_ratio=0.05):
    """Return a padded [min, max] range for axis limits."""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        pad = max(0.5, abs(vmin) * 0.1)
        return (vmin - pad, vmax + pad)
    span = vmax - vmin
    pad = span * pad_ratio
    return (vmin - pad, vmax + pad)


def compute_shared_axis_limits(raw_points_by_type_list, x_getter, y_getter, surface_mask_fn=None, show_scatter=True):
    """Compute shared x/y/z limits per metric across multiple datasets.
    
    If show_scatter=False, uses only averaged z values (from surface interpolation)
    rather than raw scatter points for computing z-axis limits.
    """
    limits_by_field = {}

    for field, _ in PLOT_SPECS:
        xs_all = []
        ys_all = []
        zs_all = []

        for raw_points_by_type in raw_points_by_type_list:
            for points in raw_points_by_type.values():
                xs = np.array([x_getter(p) for p in points], dtype=float)
                ys = np.array([y_getter(p) for p in points], dtype=float)
                zs = np.array([p[field] for p in points], dtype=float)

                valid = ~np.isnan(zs)
                xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
                
                # Apply logarithmic transformation to matching time
                if field == 'time':
                    positive_mask = zs_v > 0
                    xs_v = xs_v[positive_mask]
                    ys_v = ys_v[positive_mask]
                    zs_v = np.log10(zs_v[positive_mask])
                
                if len(xs_v) == 0:
                    continue

                if surface_mask_fn is not None:
                    domain_mask = surface_mask_fn(xs_v, ys_v)
                    xs_v = xs_v[domain_mask]
                    ys_v = ys_v[domain_mask]
                    zs_v = zs_v[domain_mask]

                if len(xs_v) == 0:
                    continue

                # If scatter is not shown, use only averaged z values for axis limits
                if not show_scatter and len(xs_v) > 0:
                    xy_to_zs = defaultdict(list)
                    for x, y, z in zip(xs_v, ys_v, zs_v):
                        xy_to_zs[(x, y)].append(z)
                    xs_v = np.array([k[0] for k in xy_to_zs])
                    ys_v = np.array([k[1] for k in xy_to_zs])
                    zs_v = np.array([np.mean(v) for v in xy_to_zs.values()])

                xs_all.extend(xs_v.tolist())
                ys_all.extend(ys_v.tolist())
                zs_all.extend(zs_v.tolist())

        if xs_all and ys_all and zs_all:
            limits_by_field[field] = {
                'xlim': _expand_axis_range(np.array(xs_all, dtype=float)),
                'ylim': _expand_axis_range(np.array(ys_all, dtype=float)),
                'zlim': _expand_axis_range(np.array(zs_all, dtype=float)),
            }

    return limits_by_field


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
                      x_getter, y_getter, save_path, show_scatter=True,
                      surface_mask_fn=None, view_elev=24, view_azim=225,
                      shared_axis_limits=None,
                      highlight_obj0_line=False,
                      highlight_num_solutions_eq1=False,
                      apply_visual_scale=True):
    """Shared helper: create a 3x3 figure of 3D surface+scatter subplots.

    x_getter / y_getter are callables that accept a raw point dict and return
    the X or Y coordinate for that point.
    show_scatter: if False, only the interpolated surface is drawn.
    apply_visual_scale: if True, apply VISUAL_SCALE to line widths and marker sizes.
    """
    type_colors = {
        'with_objects': '#1f77b4',
        'no_objects': '#ff7f0e',
        'unknown': '#2ca02c',
        'all': '#1f77b4',
    }
    type_labels_map = {
        'with_objects': 'With Objects',
        'no_objects': 'Without Objects',
        'unknown': 'Unknown',
    }
    plot_specs = PLOT_SPECS

    fig = plt.figure(figsize=(42, 30))
    fig.patch.set_facecolor('white')
    fig.suptitle(fig_title, fontsize=_fs(16))
    experiment_types = sorted(raw_points_by_type.keys())

    for plot_idx, (field, zlabel) in enumerate(plot_specs):
        ax = fig.add_subplot(3, 3, plot_idx + 1, projection='3d')
        ax.patch.set_facecolor('white')
        # Use a back-side camera angle so axis tags sit behind the plotted data.
        ax.view_init(elev=view_elev, azim=view_azim)
        
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

        obj0_label_added = False
        solutions1_label_added = False

        for exp_type in experiment_types:
            points = raw_points_by_type[exp_type]
            color = type_colors.get(exp_type, '#2ca02c')
            label = type_labels_map.get(exp_type, exp_type)

            xs = np.array([x_getter(p) for p in points], dtype=float)
            ys = np.array([y_getter(p) for p in points], dtype=float)
            zs = np.array([p[field] for p in points], dtype=float)

            valid = ~np.isnan(zs)
            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
            
            # Apply logarithmic transformation to matching time
            if field == 'time':
                # Filter out zero/negative values before log transformation
                positive_mask = zs_v > 0
                xs_v = xs_v[positive_mask]
                ys_v = ys_v[positive_mask]
                zs_v = np.log10(zs_v[positive_mask])

            # Restrict plotted data to the allowed domain when requested
            # (e.g., only #S-Rooms <= #A-Rooms for absolute-room plots).
            if surface_mask_fn is not None:
                domain_mask_points = surface_mask_fn(xs_v, ys_v)
                xs_v = xs_v[domain_mask_points]
                ys_v = ys_v[domain_mask_points]
                zs_v = zs_v[domain_mask_points]

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

                    if surface_mask_fn is not None:
                        domain_mask = surface_mask_fn(Xi, Yi)
                    else:
                        domain_mask = np.ones_like(Xi, dtype=bool)

                    # Fill interpolation holes inside the allowed domain so
                    # surfaces don't show large blank regions in --no-scatter mode.
                    if np.any(np.isnan(Zi) & domain_mask):
                        Zi_nearest = griddata((xs_agg, ys_agg), zs_agg, (Xi, Yi), method='nearest')
                        Zi = np.where(np.isnan(Zi) & domain_mask, Zi_nearest, Zi)

                    Zi = np.where(domain_mask, Zi, np.nan)
                    ax.plot_surface(Xi, Yi, Zi, alpha=0.35, color=color,
                                    linewidth=0, antialiased=True)
                    surface_drawn = True

                    # Emphasize num_solutions==1 contour when requested.
                    if field == 'num_solutions' and highlight_num_solutions_eq1:
                        try:
                            contour_level = 1.0
                            contours = ax.contour(
                                Xi,
                                Yi,
                                Zi,
                                levels=[contour_level],
                                colors=[color],
                                linewidths=6.0 * (3.0 if apply_visual_scale else 1.0),
                                alpha=1.0,
                            )
                            for collection in contours.collections:
                                for path in collection.get_paths():
                                    vertices = path.vertices
                                    if len(vertices) > 1:
                                        # Draw dark base stroke + surface color top stroke for contrast.
                                        ax.plot(
                                            vertices[:, 0],
                                            vertices[:, 1],
                                            [contour_level] * len(vertices),
                                            color='#000000',
                                            linewidth=6.5 * (3.0 if apply_visual_scale else 1.0),
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            vertices[:, 0],
                                            vertices[:, 1],
                                            [contour_level] * len(vertices),
                                            color=color,
                                            linewidth=4.5 * (3.0 if apply_visual_scale else 1.0),
                                            alpha=1.0,
                                            label='num_solutions == 1' if not solutions1_label_added else None,
                                        )
                                        solutions1_label_added = True
                        except Exception:
                            pass
                except Exception:
                    pass

            if show_scatter:
                ax.scatter(xs_v, ys_v, zs_v, color=color, s=20 * (3.0 if apply_visual_scale else 1.0),
                           label=label, depthshade=True)
            elif not surface_drawn:
                # Surface interpolation can fail for sparse/degenerate point sets.
                # In no-scatter mode, draw a minimal fallback so plots are not blank.
                ax.scatter(xs_v, ys_v, zs_v, color=color, s=10 * (3.0 if apply_visual_scale else 1.0),
                           label=label, depthshade=True, alpha=0.8)
            elif plot_idx == 0:
                # Still need a handle for the legend when scatter is hidden
                ax.scatter([], [], [], color=color, s=20 * (3.0 if apply_visual_scale else 1.0), label=label)

            # Highlight %objects==0 trace for normalized plots.
            if highlight_obj0_line:
                obj0_mask = np.isclose(ys_v, 0.0)
                if np.count_nonzero(obj0_mask) >= 2:
                    xs_obj0 = xs_v[obj0_mask]
                    zs_obj0 = zs_v[obj0_mask]

                    # Average duplicate x locations to keep the line readable.
                    x_to_z = defaultdict(list)
                    for x_val, z_val in zip(xs_obj0, zs_obj0):
                        x_to_z[float(x_val)].append(float(z_val))

                    x_sorted = np.array(sorted(x_to_z.keys()), dtype=float)
                    z_sorted = np.array([np.mean(x_to_z[x]) for x in x_sorted], dtype=float)
                    y_sorted = np.zeros_like(x_sorted)

                    if len(x_sorted) >= 2:
                        # Draw a dark base stroke + orange top stroke for high visibility.
                        ax.plot(
                            x_sorted,
                            y_sorted,
                            z_sorted,
                            color='#111111',
                            linewidth=7.0 * (3.0 if apply_visual_scale else 1.0),
                            alpha=0.95,
                            solid_capstyle='round',
                        )
                        ax.plot(
                            x_sorted,
                            y_sorted,
                            z_sorted,
                            color='#ff7a00',
                            linewidth=5.0 * (3.0 if apply_visual_scale else 1.0),
                            alpha=1.0,
                            solid_capstyle='round',
                            label='%objects == 0' if not obj0_label_added else None,
                        )
                        obj0_label_added = True

        ax.set_xlabel(xlabel, fontsize=_fs(12), labelpad=_fs(6))
        ax.set_ylabel(ylabel, fontsize=_fs(12), labelpad=_fs(6))
        # Append (log10) to zlabel for time plots
        zlabel_display = zlabel if field != 'time' else f"{zlabel} (log10)"
        ax.set_zlabel(zlabel_display, fontsize=_fs(12), labelpad=_fs(6))
        ax.set_title(zlabel_display, fontsize=_fs(13), fontweight='bold', pad=_fs(3))
        ax.tick_params(axis='x', which='major', labelsize=_fs(10), pad=_fs(2))
        ax.tick_params(axis='y', which='major', labelsize=_fs(10), pad=_fs(2))
        ax.tick_params(axis='z', which='major', labelsize=_fs(10), pad=_fs(2))

        if shared_axis_limits and field in shared_axis_limits:
            ax.set_xlim(*shared_axis_limits[field]['xlim'])
            ax.set_ylim(*shared_axis_limits[field]['ylim'])
            ax.set_zlim(*shared_axis_limits[field]['zlim'])

        # Keep tick count low to avoid label collisions with large text.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        
        # For num_solutions plot, use integer-only z-axis ticks
        if field == 'num_solutions':
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        
        if plot_idx == 0:
            ax.legend(fontsize=_fs(8.5), loc='upper center',
                      bbox_to_anchor=(0.5, 1.18), frameon=True)

    # Explicit spacing works better than tight_layout for dense 3D grids.
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.92,
                        wspace=0.34, hspace=0.36)
    return fig


def _slugify(text):
    """Convert text into a filesystem-safe lowercase slug."""
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower())
    return cleaned.strip('_') or 'subplot'


def save_individual_subplots(fig, output_dir):
    """Save each subplot from a figure as an individual PNG image."""
    os.makedirs(output_dir, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for i, ax in enumerate(fig.get_axes(), start=1):
        if not ax.get_visible():
            continue

        title = ax.get_title().strip()
        if not title:
            title = 'Experiment Summary' if i == 9 else f'Subplot {i}'

        file_name = f"{_slugify(title)}.png"
        save_path = os.path.join(output_dir, file_name)

        # Crop from the full rendered figure to this subplot area.
        bbox = ax.get_tightbbox(renderer).expanded(1.08, 1.12)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(save_path, dpi=380, bbox_inches=bbox_inches, facecolor='white')


def create_metric_comparison_grids(json_filepaths, results_dir, save_only=False):
    """Create 2 comparison grids from already-saved subplot images.

    Output images:
    - comparison_grid_solutions.png
    - comparison_grid_matching_time.png

    Grid layout for each output:
    - Columns: one per JSON file
    - Rows: figure1_absolute_rooms (top), figure2_normalized (bottom)
    
    Returns:
    - List of figure objects if save_only=False, else None
    """
    if not json_filepaths:
        return [] if not save_only else None

    plot_types = [
        ('figure1_absolute_rooms', 'Figure 1: Absolute Rooms'),
        ('figure2_normalized', 'Figure 2: Normalized'),
    ]
    metric_specs = [
        ('solutions', '#Solutions', 'comparison_grid_solutions.png'),
        ('matching_time_s', 'Matching Time (s)', 'comparison_grid_matching_time.png'),
    ]

    json_names = [os.path.splitext(os.path.basename(p))[0] for p in json_filepaths]
    n_cols = len(json_names)
    
    # Use smaller figures for interactive display, larger for saved output
    fig_width = max(8, 4 * n_cols) if not save_only else max(8, 6 * n_cols)
    fig_height = 6 if not save_only else 10
    
    comparison_figs = []

    for metric_slug, metric_label, output_filename in metric_specs:
        fig, axes = plt.subplots(
            nrows=len(plot_types),
            ncols=n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        fig.suptitle(
            f'{metric_label} Comparison Across JSON Files',
            fontsize=14 if not save_only else 18,
            fontweight='bold',
        )

        for col, json_name in enumerate(json_names):
            for row, (plot_dir, row_title) in enumerate(plot_types):
                ax = axes[row, col]
                image_path = os.path.join(results_dir, json_name, plot_dir, f'{metric_slug}.png')

                if os.path.exists(image_path):
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.set_facecolor('#f5f5f5')
                    ax.text(
                        0.5,
                        0.5,
                        f'Missing image\n{os.path.basename(image_path)}',
                        ha='center',
                        va='center',
                        fontsize=8 if not save_only else 10,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                if row == 0:
                    ax.set_title(json_name, fontsize=10 if not save_only else 12, pad=8)
                if col == 0:
                    ax.set_ylabel(row_title, fontsize=10 if not save_only else 12, rotation=90, labelpad=12)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Only save to disk, don't load the saved image again for display
        if save_only:
            output_path = os.path.join(results_dir, output_filename)
            fig.savefig(output_path, dpi=220, bbox_inches='tight', facecolor='white')
            print(f'Comparison grid saved to: {output_path}')
            plt.close(fig)
        else:
            comparison_figs.append(fig)
    
    return comparison_figs if not save_only else None


def create_3d_plots(raw_points_by_type, metadata, save_path=None, show_scatter=True,
                    shared_axis_limits=None, apply_visual_scale=True):
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
        surface_mask_fn=lambda Xi, Yi: Xi <= (Yi + 1e-9),
        # Perspective from min #A-Rooms / max #S-Rooms corner.
        view_elev=24,
        view_azim=-45,
        shared_axis_limits=shared_axis_limits,
        highlight_num_solutions_eq1=True,
        apply_visual_scale=apply_visual_scale,
    )

    # Summary info in the 9th cell
    ax_info = fig.add_subplot(3, 3, 9)
    ax_info.patch.set_facecolor('white')
    ax_info.axis('off')
    ax_info.text(0.05, 0.95, 'Experiment Summary', transform=ax_info.transAxes,
                 fontsize=_fs(13), weight='bold', va='top')
    ax_info.text(0.05, 0.82, f"Total: {metadata['total_experiments']}",
                 transform=ax_info.transAxes, fontsize=_fs(10), va='top')
    ax_info.text(0.05, 0.70, f"Dataset:\n{metadata['dataset_file']}",
                 transform=ax_info.transAxes, fontsize=_fs(9), va='top')
    ax_info.text(0.05, 0.50, f"Generated:\n{metadata['timestamp'][:19]}",
                 transform=ax_info.transAxes, fontsize=_fs(10), va='top')

    _attach_expand_on_dblclick(fig)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"3D plot (abs rooms) saved to: {save_path}")

    return fig


def create_3d_normalized_plots(raw_points_by_type, metadata, save_path=None,
                               show_scatter=True, shared_axis_limits=None,
                               pct_obj_max_cap=None, apply_visual_scale=True):
    """3D surface plots: X = n_rooms_s/n_rooms_a ratio, Y = pct_object_nodes, Z = metric.

    All experiment types are merged into one series so that no_objects (pct_obj=0)
    and with_objects (pct_obj>0) appear as different regions of the same surface.
    """
    merged = {'all': [p for pts in raw_points_by_type.values() for p in pts]}
    if pct_obj_max_cap is None:
        norm_surface_mask = None
    else:
        norm_surface_mask = lambda Xi, Yi: Yi <= (pct_obj_max_cap + 1e-9)

    fig = _make_3d_subplots(
        merged,
        fig_title='Graph Matching: 3D View  (S/A Ratio \u00d7 % Object Nodes)',
        xlabel='S/A Ratio',
        ylabel='% Obj Nodes',
        x_getter=lambda p: p['n_rooms_s'] / max(p['n_rooms_a'], 1),
        y_getter=lambda p: p['pct_obj'],
        save_path=None,
        show_scatter=show_scatter,
        surface_mask_fn=norm_surface_mask,
        # Perspective from max S/A ratio / max %objects corner.
        view_elev=24,
        view_azim=45,
        shared_axis_limits=shared_axis_limits,
        highlight_obj0_line=True,
        highlight_num_solutions_eq1=True,
        apply_visual_scale=apply_visual_scale,
    )

    ax_info = fig.add_subplot(3, 3, 9)
    ax_info.patch.set_facecolor('white')
    ax_info.axis('off')
    ax_info.text(0.05, 0.95, 'Experiment Summary', transform=ax_info.transAxes,
                 fontsize=_fs(13), weight='bold', va='top')
    ax_info.text(0.05, 0.82, f"Total: {metadata['total_experiments']}",
                 transform=ax_info.transAxes, fontsize=_fs(10), va='top')
    ax_info.text(0.05, 0.70, f"Dataset:\n{metadata['dataset_file']}",
                 transform=ax_info.transAxes, fontsize=_fs(9), va='top')
    ax_info.text(0.05, 0.50, f"Generated:\n{metadata['timestamp'][:19]}",
                 transform=ax_info.transAxes, fontsize=_fs(10), va='top')

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
    parser.add_argument('--json-files', nargs='+', default=None,
                       help='One or more JSON files to process with shared axis scales')
    parser.add_argument('--display', action='store_true',
                       help='Display plots in interactive windows (default: save only)')
    parser.add_argument('--no-scatter', action='store_true',
                       help='Hide individual data-point spheres; show surfaces only')

    args = parser.parse_args()
    
    # Switch to interactive backend only if display is explicitly requested
    if args.display:
        import matplotlib
        matplotlib.use('TkAgg')
    
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    
    requested_files = list(args.json_files) if args.json_files else None
    if requested_files is None:
        requested_file = args.json_file_opt
        if requested_file is None and args.json_file != 'latest_results.json':
            # Preserve compatibility for explicit positional usage.
            requested_file = args.json_file
        if requested_file:
            requested_files = [requested_file]

    json_filepaths = select_results_files(results_dir, requested_files)

    datasets = []
    for json_filepath in json_filepaths:
        print(f"Loading results from: {json_filepath}")
        data = load_results(json_filepath)
        if data is None:
            print(f"Skipping unreadable file: {json_filepath}")
            continue

        print(f"Loaded {len(data['experiments'])} experiments")
        (metrics_by_rooms_and_type, success_data, timing_data,
         solution_count_data, metadata, raw_points) = process_data(data)
        room_counts, metric_names, metric_labels, stats_by_type, experiment_types = calculate_statistics(
            metrics_by_rooms_and_type, success_data, timing_data, solution_count_data)

        datasets.append({
            'json_filepath': json_filepath,
            'metadata': metadata,
            'raw_points': raw_points,
            'room_counts': room_counts,
            'metric_names': metric_names,
            'metric_labels': metric_labels,
            'stats_by_type': stats_by_type,
            'experiment_types': experiment_types,
            'metrics_by_rooms_and_type': metrics_by_rooms_and_type,
        })

    if not datasets:
        print('No valid JSON datasets loaded.')
        sys.exit(1)

    show_scatter = not args.no_scatter

    abs_axis_limits = compute_shared_axis_limits(
        [d['raw_points'] for d in datasets],
        x_getter=lambda p: p['n_rooms_s'],
        y_getter=lambda p: p['n_rooms_a'],
        surface_mask_fn=lambda Xi, Yi: Xi <= (Yi + 1e-9),
        show_scatter=show_scatter,
    )

    normalized_raw_sets = [
        {'all': [p for pts in d['raw_points'].values() for p in pts]}
        for d in datasets
    ]
    norm_axis_limits = compute_shared_axis_limits(
        normalized_raw_sets,
        x_getter=lambda p: p['n_rooms_s'] / max(p['n_rooms_a'], 1),
        y_getter=lambda p: p['pct_obj'],
        surface_mask_fn=None,
        show_scatter=show_scatter,
    )

    # Cap normalized %objects axis to the minimum available upper bound across datasets
    # so all plots share the same comparable y-domain intersection.
    dataset_pct_obj_max = []
    for dataset in datasets:
        merged_points = [p for pts in dataset['raw_points'].values() for p in pts]
        if not merged_points:
            continue
        pct_vals = np.array([p['pct_obj'] for p in merged_points], dtype=float)
        if len(pct_vals) == 0:
            continue
        dataset_pct_obj_max.append(float(np.nanmax(pct_vals)))

    pct_obj_max_cap = min(dataset_pct_obj_max) if dataset_pct_obj_max else None
    if pct_obj_max_cap is not None:
        for field_limits in norm_axis_limits.values():
            y_min, y_max = field_limits['ylim']
            clipped_y_max = min(y_max, pct_obj_max_cap)
            if clipped_y_max <= y_min:
                # Keep a non-degenerate axis when all points collapse to one value.
                clipped_y_max = y_min + 1.0
            field_limits['ylim'] = (y_min, clipped_y_max)

    generated_figs = []
    for dataset in datasets:
        json_filepath = dataset['json_filepath']
        metadata = dataset['metadata']
        raw_points = dataset['raw_points']

        # Figure 1 – absolute room counts (X=S-rooms, Y=A-rooms, Z=metric)
        fig1 = create_3d_plots(
            raw_points,
            metadata,
            save_path=None,
            show_scatter=show_scatter,
            shared_axis_limits=abs_axis_limits,
            apply_visual_scale=False,  # Don't scale for disk output
        )

        # Figure 2 – normalised axes (X=S/A ratio, Y=%obj nodes, Z=metric)
        fig2 = create_3d_normalized_plots(
            raw_points,
            metadata,
            save_path=None,
            show_scatter=show_scatter,
            shared_axis_limits=norm_axis_limits,
            pct_obj_max_cap=pct_obj_max_cap,
            apply_visual_scale=False,  # Don't scale for disk output
        )

        generated_figs.extend([fig1, fig2])

        # Save each subplot as an independent image:
        # results/<json_file>/<plot_type>/<subplot_name>.png
        json_name = os.path.splitext(os.path.basename(json_filepath))[0]
        output_root = os.path.join(results_dir, json_name)
        abs_dir = os.path.join(output_root, 'figure1_absolute_rooms')
        norm_dir = os.path.join(output_root, 'figure2_normalized')
        save_individual_subplots(fig1, abs_dir)
        save_individual_subplots(fig2, norm_dir)
        print(f"Saved individual subplots to: {output_root}")

        # Print numerical summary
        print_numerical_summary(
            dataset['room_counts'],
            dataset['metric_names'],
            dataset['metric_labels'],
            dataset['stats_by_type'],
            dataset['experiment_types'],
            dataset['metrics_by_rooms_and_type'],
        )
    
    # Display plots unless disabled
    if args.display:
        # Close large-text figures now that images are saved
        for fig in generated_figs:
            plt.close(fig)
        
        # Reset text and visual scales for normal-sized interactive display
        global TEXT_SCALE, VISUAL_SCALE
        TEXT_SCALE = 1.0
        VISUAL_SCALE = 1.0
        generated_figs = []
        
        # Recreate figures with normal font sizes but scaled visual elements for display
        for dataset in datasets:
            json_filepath = dataset['json_filepath']
            metadata = dataset['metadata']
            raw_points = dataset['raw_points']
            
            fig1 = create_3d_plots(
                raw_points,
                metadata,
                save_path=None,
                show_scatter=show_scatter,
                shared_axis_limits=abs_axis_limits,
                apply_visual_scale=True,  # Scale line widths for display
            )
            
            fig2 = create_3d_normalized_plots(
                raw_points,
                metadata,
                save_path=None,
                show_scatter=show_scatter,
                shared_axis_limits=norm_axis_limits,
                pct_obj_max_cap=pct_obj_max_cap,
                apply_visual_scale=True,  # Scale line widths for display
            )
            
            generated_figs.extend([fig1, fig2])
        
        # Create and display comparison grids with normal sizing
        comparison_figs = create_metric_comparison_grids(json_filepaths, results_dir, save_only=False)
        generated_figs.extend(comparison_figs)
        
        plt.show()
    else:
        for fig in generated_figs:
            plt.close(fig)
        print("Plots saved to disk. Use --display flag to view in interactive windows.")
    
    # Build comparison grids and save to disk (if not already saved)
    create_metric_comparison_grids(json_filepaths, results_dir, save_only=True)
    
    print(f"\nResults processed successfully!")
    print("  - Images saved to disk")
    print("  - Use --display flag to open windows interactively")


if __name__ == "__main__":
    main()