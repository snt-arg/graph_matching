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
    
    for experiment in experiments:
        n_rooms = experiment['n_rooms_s_graphs']
        exp_type = experiment.get('experiment_type', 'unknown')
        
        # Extract matching time
        matching_time = experiment.get('matching_time', 0.0)
        timing_data[exp_type][n_rooms].append(matching_time)
        
        # Check success: use explicit 'success' field if present, else infer from metrics
        if 'success' in experiment:
            is_success = experiment['success']
        else:
            is_success = 'metrics' in experiment and bool(experiment['metrics'])
        success_data[exp_type][n_rooms].append(1 if is_success else 0)
        
        has_metrics = ('metrics' in experiment and 
                       bool(experiment['metrics']) and 
                       'precision' in experiment['metrics'])
        if has_metrics:
            metrics = experiment['metrics']
            metrics_by_rooms_and_type[exp_type][n_rooms].append(metrics)

        # num_solutions is always at the top level of the experiment, not inside metrics
        num_solutions = experiment.get('num_solutions', 0)
        solution_count_data[exp_type][n_rooms].append(num_solutions)
    
    return metrics_by_rooms_and_type, success_data, timing_data, solution_count_data, metadata


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


def create_plots(room_counts, metric_names, metric_labels, stats_by_type, 
                experiment_types, metadata, save_path=None):
    """Create comprehensive subplot visualization comparing experiment types."""
    
    # Create larger subplot visualization (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle('Graph Matching Performance: With vs Without Objects', fontsize=20, y=0.98)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Color scheme for each experiment type
    type_colors = {'with_objects': '#1f77b4', 'no_objects': '#ff7f0e', 'unknown': '#2ca02c'}
    type_labels = {'with_objects': 'With Objects', 'no_objects': 'Without Objects', 'unknown': 'Unknown'}
    
    # X-offsets so overlapping series remain visible
    type_offsets = {'with_objects': -0.03, 'no_objects': 0.03, 'unknown': 0.0}

    # Plot performance metrics (first 5 plots)
    for metric_idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes_flat[metric_idx]
        
        # Plot each experiment type
        for exp_type in experiment_types:
            if exp_type in stats_by_type:
                averaged_metrics = stats_by_type[exp_type]['averaged_metrics']
                std_metrics = stats_by_type[exp_type]['std_metrics']
                x_offset = type_offsets.get(exp_type, 0.0)
                
                # Filter out NaN values — use vi (not i) to avoid shadowing outer loop var
                valid_indices = [vi for vi, val in enumerate(averaged_metrics[metric]) if not np.isnan(val)]
                valid_room_counts = [room_counts[vi] + x_offset for vi in valid_indices]
                valid_means = [averaged_metrics[metric][vi] for vi in valid_indices]
                valid_stds = [std_metrics[metric][vi] for vi in valid_indices]
                
                if valid_means:  # Only plot if we have valid data
                    ax.errorbar(valid_room_counts, valid_means, 
                               yerr=valid_stds, 
                               marker='o', linestyle='-', linewidth=2.5, markersize=8,
                               capsize=4, capthick=1.5, 
                               color=type_colors.get(exp_type, '#2ca02c'),
                               label=type_labels.get(exp_type, exp_type.title()))
        
        ax.set_xlabel('Number of Rooms (S-Graph)', fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.set_title(f'{label} vs Number of Rooms', fontsize=11, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        if metric_idx == 0:  # Only show legend on first plot
            ax.legend(fontsize=9, loc='lower left')
    
    # Plot success rates (6th plot)
    ax_success = axes_flat[5]
    for exp_type in experiment_types:
        if exp_type in stats_by_type:
            success_rates = stats_by_type[exp_type]['success_rates']
            
            x_offset = type_offsets.get(exp_type, 0.0)
            valid_indices = [vi for vi, val in enumerate(success_rates) if not np.isnan(val)]
            valid_room_counts = [room_counts[vi] + x_offset for vi in valid_indices]
            valid_rates = [success_rates[vi] for vi in valid_indices]
            
            if valid_rates:
                ax_success.plot(valid_room_counts, valid_rates,
                               marker='o', linestyle='-', linewidth=2.5, markersize=8,
                               color=type_colors.get(exp_type, '#2ca02c'),
                               label=type_labels.get(exp_type, exp_type.title()))
    
    ax_success.set_xlabel('Number of Rooms (S-Graph)', fontsize=10, fontweight='bold')
    ax_success.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax_success.set_title('Success Rate vs Number of Rooms', fontsize=11, fontweight='bold', pad=15)
    ax_success.grid(True, alpha=0.3, linestyle='--')
    ax_success.set_xlim(left=0)
    ax_success.set_ylim(0, 105)
    ax_success.legend(fontsize=9, loc='lower right')
    
    # Plot average matching times (7th plot)
    ax_time = axes_flat[6]
    for exp_type in experiment_types:
        if exp_type in stats_by_type:
            avg_times = stats_by_type[exp_type]['avg_times']
            std_times = stats_by_type[exp_type]['std_times']
            
            x_offset = type_offsets.get(exp_type, 0.0)
            valid_indices = [vi for vi, val in enumerate(avg_times) if not np.isnan(val)]
            valid_room_counts = [room_counts[vi] + x_offset for vi in valid_indices]
            valid_times = [avg_times[vi] for vi in valid_indices]
            valid_time_stds = [std_times[vi] for vi in valid_indices]
            
            if valid_times:
                ax_time.errorbar(valid_room_counts, valid_times,
                                yerr=valid_time_stds,
                                marker='o', linestyle='-', linewidth=2.5, markersize=8,
                                capsize=4, capthick=1.5,
                                color=type_colors.get(exp_type, '#2ca02c'),
                                label=type_labels.get(exp_type, exp_type.title()))
    
    ax_time.set_xlabel('Number of Rooms (S-Graph)', fontsize=10, fontweight='bold')
    ax_time.set_ylabel('Matching Time (seconds)', fontsize=10, fontweight='bold')
    ax_time.set_title('Matching Time vs Number of Rooms', fontsize=11, fontweight='bold', pad=15)
    ax_time.set_yscale('log')
    ax_time.grid(True, alpha=0.3, linestyle='--', which='both')
    ax_time.set_xlim(left=0)
    ax_time.legend(fontsize=9, loc='upper left')
    
    # Plot average solution counts (8th plot)
    ax_solutions = axes_flat[7]
    for exp_type in experiment_types:
        if exp_type in stats_by_type:
            avg_solutions = stats_by_type[exp_type]['avg_solutions']
            std_solutions = stats_by_type[exp_type]['std_solutions']
            
            x_offset = type_offsets.get(exp_type, 0.0)
            valid_indices = [vi for vi, val in enumerate(avg_solutions) if not np.isnan(val)]
            valid_room_counts = [room_counts[vi] + x_offset for vi in valid_indices]
            valid_solution_counts = [avg_solutions[vi] for vi in valid_indices]
            valid_solution_stds = [std_solutions[vi] for vi in valid_indices]
            
            if valid_solution_counts:
                ax_solutions.errorbar(valid_room_counts, valid_solution_counts,
                                     yerr=valid_solution_stds,
                                     marker='o', linestyle='-', linewidth=2.5, markersize=8,
                                     capsize=4, capthick=1.5,
                                     color=type_colors.get(exp_type, '#2ca02c'),
                                     label=type_labels.get(exp_type, exp_type.title()))
    
    ax_solutions.set_xlabel('Number of Rooms (S-Graph)', fontsize=10, fontweight='bold')
    ax_solutions.set_ylabel('Number of Solutions Found', fontsize=10, fontweight='bold')
    ax_solutions.set_title('Solution Count vs Number of Rooms', fontsize=11, fontweight='bold', pad=15)
    ax_solutions.grid(True, alpha=0.3, linestyle='--')
    ax_solutions.set_xlim(left=0)
    ax_solutions.legend(fontsize=9, loc='upper right')
    
    # Add summary statistics subplot (9th plot)
    ax_summary = axes_flat[8]
    ax_summary.text(0.05, 0.95, 'Experiment Summary', 
                   transform=ax_summary.transAxes, fontsize=12, weight='bold')
    ax_summary.text(0.05, 0.85, f'Total Experiments: {metadata["total_experiments"]}', 
                   transform=ax_summary.transAxes, fontsize=10, weight='bold')
    ax_summary.text(0.05, 0.75, f'Dataset: {metadata["dataset_file"]}', 
                   transform=ax_summary.transAxes, fontsize=9)
    ax_summary.text(0.05, 0.65, f'Generated: {metadata["timestamp"][:19]}', 
                   transform=ax_summary.transAxes, fontsize=9)
    
    # Count experiments by type
    ax_summary.text(0.05, 0.50, 'Experiments by Type:', 
                   transform=ax_summary.transAxes, fontsize=10, weight='bold')
    
    line_y = 0.40
    for exp_type in experiment_types:
        type_label = type_labels.get(exp_type, exp_type.title())
        # Count total experiments for this type across all room counts
        total_exp = len(stats_by_type.get(exp_type, {}).get('success_rates', []))
        ax_summary.text(0.1, line_y, f'{type_label}: {total_exp} room configs', 
                       transform=ax_summary.transAxes, fontsize=9)
        line_y -= 0.1
    
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], pad=3.0, h_pad=3.5, w_pad=2.0)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")
    
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
    metrics_by_rooms_and_type, success_data, timing_data, solution_count_data, metadata = process_data(data)
    room_counts, metric_names, metric_labels, stats_by_type, experiment_types = calculate_statistics(
        metrics_by_rooms_and_type, success_data, timing_data, solution_count_data)
    
    # Create output filenames with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"graph_matching_plots_{timestamp}.png"
    summary_filename = f"graph_matching_summary_{timestamp}.txt"
    
    plot_path = os.path.join(results_dir, plot_filename)
    summary_path = os.path.join(results_dir, summary_filename)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots
    fig = create_plots(room_counts, metric_names, metric_labels, 
                      stats_by_type, experiment_types, metadata, plot_path)
    
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
        plt.close(fig)
        print("Plots saved but not displayed (--no-display flag used)")
    
    print(f"\nResults processed successfully!")
    print(f"  - Plot: {plot_path}")
    print(f"  - Summary: {summary_path}")


if __name__ == "__main__":
    main()