import matplotlib.pyplot as plt
import sys, json, os, copy
import numpy as np
import time
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict

from sympy import false

from GraphMatcher import GraphMatcher
from utils import plane_4_params_to_6_params, plane_6_params_to_4_params, correct_plane_direction


# reasoning_msgs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# with open(os.path.join(reasoning_msgs,"config", "syntheticDS_params_synthetic.json")) as f:
#     syntheticDS_params = json.load(f)

syntheticDS_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "syntheticDS_params_synthetic.json")
with open(syntheticDS_params_path) as f:
    syntheticDS_params = json.load(f)

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
# from situational_graphs_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from situational_graphs_datasets.graph_visualizer import visualize_nxgraph_3d

import situational_graphs_wrapper
# Make the module available under both names for pickle compatibility
sys.modules['graph_wrapper'] = situational_graphs_wrapper

# from situational_graphs_datasets.config import get_config as get_datasets_config
# synteticdataset_settings = get_datasets_config("graph_matching")

class FakeLogger(object):
    def __init__(self) -> None:
        pass
    def info(self, msg):
        print(f"FakeLogger: {msg}")
fake_logger = FakeLogger()

# # GENERATE DATASET

# dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
# dataset_generator.create_dataset()

# ### A-GRAPH
# a_dataset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")["train"]
# visualize_nxgraph_3d(a_dataset[0], "a_graph", visualize_alone=True, include_node_ids=True)
# # for node_attrs in a_dataset[0].get_attributes_of_all_nodes():
# #     print(node_attrs[1]["Geometric_info"])

# ### S-GRAPH
# s_dataset = copy.deepcopy(a_dataset)

# postprocess = [{"pp_name": "dropout", "room": 0.9, "ws":0.0}]
# dataset_generator.settings["postprocess"]["sgraph"] = postprocess
# s_dataset = dataset_generator.extend_nxdataset(copy.deepcopy(s_dataset), "training", "sgraph")["train"]

# [s_graph.translate_geometries([10,20,0]) for s_graph in s_dataset]
# visualize_nxgraph_3d(s_dataset[0], "s_graph", visualize_alone=True, include_node_ids=True)

def process_synthetic_graph(graph):
    graph.to_undirected()
    for node_id, node_attrs in graph.get_attributes_of_all_nodes():
        if node_attrs["type"] == "room":
            node_attrs["Geometric_info"] = np.array(node_attrs["center"])
        elif node_attrs["type"] == "window" or node_attrs["type"] == "door" \
            or node_attrs["type"] == "table" or node_attrs["type"] == "chair":
            node_attrs["Geometric_info"] = np.array(node_attrs["center"])
        elif node_attrs["type"] == "ws":
            initial = [node_attrs["center"][0], node_attrs["center"][1], node_attrs["center"][2],
                       node_attrs["normal"][0], node_attrs["normal"][1], node_attrs["normal"][2]]

            # node_attrs["Geometric_info"] = plane_4_params_to_6_params(plane_6_params_to_4_params(initial))
            node_attrs["Geometric_info"] = initial
            

def run_graph_matching_experiment(a_graph, s_graph, gt_match, graph_name_suffix=""):
    """
    Run a single graph matching experiment and return results.
    
    Args:
        a_graph: The reference (A) graph
        s_graph: The target (S) graph  
        graph_name_suffix: Suffix to add to graph names for identification

    
    Returns:
        dict: Dictionary containing success status, metrics, timing, and other results
    """
    # Prepare target graph (with or without filtering)
    target_graph = copy.deepcopy(s_graph)
    
    # Create GraphMatcher
    graph_matcher = GraphMatcher(fake_logger, log_level=0)
    graph_matcher.set_parameters(syntheticDS_params)
    graph_matcher.set_graph_from_wrapper(a_graph, "A-Graph")
    graph_matcher.set_graph_from_wrapper(s_graph, f"S-Graph")
    graph_matcher.room_string = "room"
    graph_matcher.ws_string = "ws"
    
    # Perform matching
    start_time = time.time()
    success, matches, matches_full, matches_dev = graph_matcher.match("A-Graph", f"S-Graph")
    matching_time = time.time() - start_time
    
    # Process results - use matches (all candidates) instead of matches_full
    # matches_full is only populated for unique matches (len==1), so using matches
    # correctly reflects all candidates including symmetric/ambiguous cases.
    matches_node_ids = []
    for final_combination in matches:
        node_ids = []
        for pair in final_combination:
            node_ids.append([str(pair['origin_node']), str(pair['target_node'])])
        matches_node_ids.append(node_ids)
    
    # Compute metrics only when there is exactly one unambiguous solution.
    # Multiple solutions mean symmetry ambiguity - success/time/num_solutions are
    # still recorded, but classification metrics are not meaningful in that case.
    metrics = None
    if success and len(matches_node_ids) == 1:
        metrics = compute_metrics(gt_match, matches_node_ids[0])
    
    print(f"dbg matches_node_ids {matches_node_ids}")
    return {
        'success': success,
        'matches': matches,
        'matches_full': matches_full,
        'matches_node_ids': matches_node_ids,
        'metrics': metrics,
        'matching_time': matching_time,
        'gt_match': gt_match,
        'experiment_type': graph_name_suffix
    }


def compute_metrics(ground_truth_matches, predicted_matches):
    """
    Compute graph matching performance metrics treating it as node correspondence classification.
    
    Args:
        ground_truth_matches: List of ground truth matches [[node_A, node_B], ...]
                              where node_A corresponds to node_B
        predicted_matches: List of predicted matches [[node_A, node_B], ...]
                          where node_A corresponds to node_B
    
    Returns:
        dict: Dictionary containing confusion matrix and performance metrics
    """
    
    # print(f"\nDEBUG: ground_truth_matches = {ground_truth_matches}")
    # print(f"DEBUG: predicted_matches = {predicted_matches}")
    
    # Convert matches to sets of tuples for easier comparison
    # For graph matching, we keep the order (node_A, node_B) - don't sort!
    gt_set = set()
    pred_set = set()
    
    for match in ground_truth_matches:
        correspondence = (str(match[0]), str(match[1]))
        gt_set.add(correspondence)
    
    for match in predicted_matches:
        correspondence = (str(match[0]), str(match[1]))
        pred_set.add(correspondence)
    
    # print(f"DEBUG: gt_set = {gt_set}")
    # print(f"DEBUG: pred_set = {pred_set}")
    
    # Get all unique nodes from both graphs
    nodes_A = set()  # nodes from first graph
    nodes_B = set()  # nodes from second graph
    
    for match in ground_truth_matches + predicted_matches:
        nodes_A.add(str(match[0]))
        nodes_B.add(str(match[1]))
    
    nodes_A = list(nodes_A)
    nodes_B = list(nodes_B)
    
    # print(f"DEBUG: nodes_A = {sorted(nodes_A)}")
    # print(f"DEBUG: nodes_B = {sorted(nodes_B)}")
    
    # Generate all possible correspondences (node_A, node_B)
    all_possible_correspondences = set()
    for node_a in nodes_A:
        for node_b in nodes_B:
            correspondence = (node_a, node_b)
            all_possible_correspondences.add(correspondence)
    
    # print(f"DEBUG: Total possible correspondences: {len(all_possible_correspondences)}")
    # print(f"DEBUG: First 5 correspondences = {list(all_possible_correspondences)[:5]}")
    
    # Create binary labels for all possible correspondences
    y_true = []
    y_pred = []
    
    for correspondence in all_possible_correspondences:
        y_true.append(1 if correspondence in gt_set else 0)
        y_pred.append(1 if correspondence in pred_set else 0)
    
    # print(f"DEBUG: y_true sum = {sum(y_true)} (should equal {len(gt_set)})")
    # print(f"DEBUG: y_pred sum = {sum(y_pred)} (should equal {len(pred_set)})")
    
    # Compute confusion matrix with explicit labels to ensure 2x2 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel()
    
    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate correct and incorrect matches
    correct_matches = len(gt_set.intersection(pred_set))
    total_gt_matches = len(gt_set)
    total_pred_matches = len(pred_set)
    
    metrics = {
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'correct_matches': correct_matches,
        'total_gt_matches': total_gt_matches,
        'total_pred_matches': total_pred_matches
    }
    
    # Print metrics
    print("\n" + "="*60)
    print("GRAPH MATCHING PERFORMANCE METRICS")
    print("="*60)
    print(f"Total possible correspondences: {len(all_possible_correspondences)}")
    print(f"Ground truth matches: {len(gt_set)}")
    print(f"Predicted matches: {len(pred_set)}")
    print(f"Correct matches: {correct_matches}")
    print(f"Match accuracy: {correct_matches/max(total_gt_matches, 1):.4f}")
    print("\nConfusion Matrix:")
    print(f"                     Predicted")
    print(f"                   No Match  Match")
    print(f"Actual  No Match   {tn:6d}  {fp:5d}")
    print(f"        Match      {fn:6d}  {tp:5d}")
    print("\nClassification Metrics:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*60)
    
    return metrics



pickle_datasets_path = "/home/adminpc/datasets_matching/pickles"
full_dataset = pickle.load(open(os.path.join(pickle_datasets_path, "incremental_symmetries_grid_squared_translation.pkl"), "rb"))

# Initialize data collection for visualization
all_metrics_data = []  # List to store (n_rooms, metrics_dict) tuples

for a_graph, s_graphs in full_dataset[3:]:
    # visualize_nxgraph_3d(a_graph, "a_graph", visualize_alone=True, include_node_ids=True)
    process_synthetic_graph(a_graph)
    for i, s_graph in enumerate(s_graphs):
        process_synthetic_graph(s_graph)
        # visualize_nxgraph_3d(s_graph, f"s_graph_{i}", visualize_alone=True, include_node_ids=True)
        # input("Press Enter to continue...")
        a_graph.name = "A-Graph"
        s_graph.name = "S-Graph"

        a_graph_nodes_ids = copy.deepcopy(a_graph.get_nodes_ids())
        a_graph.stringify_node_ids()
        mapping = dict(zip(s_graph.get_nodes_ids(), list(np.array(s_graph.get_nodes_ids()) + len(a_graph_nodes_ids) + 1)))

        n_rooms_s_graphs = len(copy.deepcopy(s_graph).filter_graph_by_node_types(["room"]).get_nodes_ids())

        s_graph.stringify_node_ids()

        # Run graph matching experiments
        print(f"Running experiment with {n_rooms_s_graphs} rooms...")

        # Create ground truth matches
        a_graph_no_objects = copy.deepcopy(a_graph).filter_graph_by_node_types(["room", "ws"])
        s_graph_no_objects = copy.deepcopy(s_graph).filter_graph_by_node_types(["room", "ws"])
        gt_match = [[i, i] for i in s_graph_no_objects.get_nodes_ids()]
        
        # Experiment 1: With all objects
        results_with_objects = run_graph_matching_experiment(
            a_graph, s_graph, gt_match,
            graph_name_suffix=f"with_objects_{n_rooms_s_graphs}_rooms"
        )

        metrics_with_objects = results_with_objects['metrics'].copy() if results_with_objects['metrics'] else {}
        metrics_with_objects['experiment_type'] = 'with_objects'
        metrics_with_objects['matching_time'] = results_with_objects['matching_time']
        metrics_with_objects['num_solutions'] = len(results_with_objects['matches_node_ids'])
        metrics_with_objects['success'] = bool(results_with_objects['success'])
        all_metrics_data.append((n_rooms_s_graphs, metrics_with_objects))
        
        # Experiment 2: Without objects (only rooms and walls)
        results_no_objects = run_graph_matching_experiment(
            a_graph_no_objects, s_graph_no_objects, gt_match,
            graph_name_suffix=f"no_objects_{n_rooms_s_graphs}_rooms"
        )

        metrics_no_objects = results_no_objects['metrics'].copy() if results_no_objects['metrics'] else {}
        metrics_no_objects['experiment_type'] = 'no_objects'
        metrics_no_objects['matching_time'] = results_no_objects['matching_time']
        metrics_no_objects['num_solutions'] = len(results_no_objects['matches_node_ids'])
        metrics_no_objects['success'] = bool(results_no_objects['success'])
        all_metrics_data.append((n_rooms_s_graphs, metrics_no_objects))
        
        # if results_with_objects['success'] and results_with_objects['metrics']:
        #     pass
            ### Plot the matched graphs (disabled for batch processing)
            # s_graph.relabel_nodes(mapping, copy=True)
            # as_graph = copy.deepcopy(a_graph)
            # as_graph.add_nodes(s_graph.get_attributes_of_all_nodes())
            # as_graph.add_edges(s_graph.get_attributes_of_all_edges())
            # edges = []
            # for edge_dict in results_with_objects['matches_full'][0]:
            #     edges.append((str(edge_dict['origin_node']), str(edge_dict['target_node']), {"viz_feat" : 'green'}))
            # as_graph.add_edges(edges)
            # visualize_nxgraph_3d(as_graph, "as_graph", visualize_alone=True, include_node_ids=True)
            # input("Press Enter to continue...")  # Disabled for batch processing


# Save collected data to JSON file
if all_metrics_data:
    import json
    import datetime
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data for JSON serialization
    # Convert numpy arrays and other non-serializable objects
    serializable_data = []
    for n_rooms, metrics in all_metrics_data:
        serializable_metrics = {}
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                serializable_metrics[key] = value.tolist()  # Convert numpy array to list
            elif isinstance(value, bool):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = float(value) if isinstance(value, np.number) else value
        
        serializable_data.append({
            'n_rooms_s_graphs': int(n_rooms),
            'metrics': serializable_metrics,
            'experiment_type': metrics.get('experiment_type', 'unknown'),
            'matching_time': metrics.get('matching_time', 0.0),
            'num_solutions': metrics.get('num_solutions', 0),
            'success': metrics.get('success', False)
        })
    
    # Create metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_experiments': len(all_metrics_data),
        'dataset_file': 'incremental_translation.pkl',
        'description': 'Graph matching performance metrics collected from synthetic dataset experiments'
    }
    
    # Combine data and metadata
    output_data = {
        'metadata': metadata,
        'experiments': serializable_data
    }
    
    # Save to JSON file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"graph_matching_results_{timestamp}.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {json_filepath}")
    print(f"Total experiments saved: {len(all_metrics_data)}")
    
    # Also save a "latest" version for easy access
    latest_filepath = os.path.join(results_dir, "latest_results.json")
    with open(latest_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Latest results also saved to: {latest_filepath}")
    
else:
    print("No successful matching results to save.")