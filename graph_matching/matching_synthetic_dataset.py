import matplotlib.pyplot as plt
import sys, json, os, copy
import numpy as np
import time
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict

# from sympy import false
from tqdm import tqdm

# ── Matcher / dataset selection ───────────────────────────────────────────────
# USE_PGM=True  → run with PGM_env (Python 3.11, has PyTorch/moviepy)
# USE_PGM=False → run with system Python 3.12 (has clipperpy/ROS packages)
USE_PGM = True

# Dataset selection — pick one: "synthetic", "msd", "real"
DATASET = "msd" 

if USE_PGM:
    import torch
    PGM_PATH = '/root/workspace/src/graph_matching_gnn/graph_matching'
    if PGM_PATH not in sys.path:
        sys.path.insert(0, PGM_PATH)
    from PGM_class import PartialGraphMatching, MatchingModel_GATv2SinkhornTopK, predict_matching_matrix  # type: ignore
else:
    from GraphMatcher import GraphMatcher


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

class FakeLogger(object):    #Needed to initialize GraphMatcher without a real logger (to avoid errors when running in non-ROS environment)
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
        elif node_attrs["type"] == "ws":
            initial = [node_attrs["center"][0], node_attrs["center"][1], node_attrs["center"][2],
                       node_attrs["normal"][0], node_attrs["normal"][1], node_attrs["normal"][2]]

            # node_attrs["Geometric_info"] = plane_4_params_to_6_params(plane_6_params_to_4_params(initial))
            node_attrs["Geometric_info"] = initial


def load_real_graphs():
    """Load Prior/Online graphs and ground truth from graph_dicts/.

    The pickled graphs are already in GNN format (split planes, sanitized attrs).
    Ground truth JSON format:
        { "rooms": {"online_id": "prior_id", ...},
          "ws":    [["online_id", "prior_id"], ...] }

    Returns:
        (a_graph, s_graph, gt_match)
        where gt_match = [[prior_node_id, online_node_id], ...] (A-graph, S-graph order)
    """
    graph_dicts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_dicts")

    with open(os.path.join(graph_dicts_path, "Prior.pkl"), 'rb') as f:
        a_graph = pickle.load(f)
    with open(os.path.join(graph_dicts_path, "Online.pkl"), 'rb') as f:
        s_graph = pickle.load(f)

    # Sanitize numpy arrays so the PGM feature builder (list concatenation) works correctly
    for g in (a_graph, s_graph):
        for _, attrs in g.graph.nodes(data=True):
            for key in ("center", "normal"):
                if key in attrs and hasattr(attrs[key], "tolist"):
                    attrs[key] = attrs[key].tolist()

    # Load and parse ground truth — store as [[prior_id, online_id]] = [[A, S]]
    gt_path = os.path.join(graph_dicts_path, "ground_truth.json")
    with open(gt_path, "r") as f:
        raw = json.load(f)

    gt_match = []
    for o_id, p_id in raw.get("rooms", {}).items():
        if p_id != "??":
            gt_match.append([str(p_id), str(o_id)])   # [prior=A, online=S]
    for entry in raw.get("ws", []):
        if len(entry) == 2 and entry[1] != "??":
            gt_match.append([str(entry[1]), str(entry[0])])  # [prior=A, online=S]

    print(f"Real graphs loaded — Prior: {a_graph.graph.number_of_nodes()} nodes, "
          f"Online: {s_graph.graph.number_of_nodes()} nodes, GT pairs: {len(gt_match)}")
    return a_graph, s_graph, gt_match
            

def run_graph_matching_experiment(a_graph, s_graph, gt_match, graph_name_suffix=""):
    """
    Run a single graph matching experiment and return results. (CLASSIC MATCHER)
    
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


def compute_metrics(ground_truth_matches, predicted_matches, log_level=0):
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
    
    if log_level >= 2:
        print(f"\nDEBUG: ground_truth_matches = {ground_truth_matches}")
        print(f"DEBUG: predicted_matches = {predicted_matches}")
    
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
    
    if log_level >= 1:
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


def run_pgm_matching_experiment(pgm_model, a_graph, s_graph, gt_match, graph_name_suffix=""):
    """
    Run a single graph matching experiment using the GNN-based PGM matcher.

    The binary matching matrix returned by infer_matching maps matrix indices to
    node IDs via list(g.nodes()).  The resulting pair list has the same format as
    run_graph_matching_experiment so compute_metrics can be reused unchanged.

    Args:
        pgm_model        : loaded PartialGraphMatching instance
        a_graph          : reference graph (situational_graphs_wrapper or NetworkX DiGraph)
        s_graph          : target graph
        gt_match         : ground truth [[node_A, node_B], ...]
        graph_name_suffix: label stored in experiment_type

    Returns:
        dict with the same keys as run_graph_matching_experiment
    """
    # infer_matching expects a raw NetworkX graph — unwrap if a GraphWrapper is passed
    g1 = a_graph.graph if hasattr(a_graph, 'graph') else a_graph
    g2 = s_graph.graph if hasattr(s_graph, 'graph') else s_graph

    start_time = time.time()
    matching_matrix = pgm_model.infer_matching(g1, g2, discrete=True)  # binary [N1, N2]
    matching_time = time.time() - start_time

    g1_nodes = list(g1.nodes())
    g2_nodes = list(g2.nodes())

    rows, cols = np.where(matching_matrix.cpu().numpy() > 0)
    predicted_matches = [[str(g1_nodes[r]), str(g2_nodes[c])] for r, c in zip(rows, cols)]

    metrics = compute_metrics(gt_match, predicted_matches)

    return {
        'success': len(predicted_matches) > 0,
        'matches_node_ids': [predicted_matches],
        'metrics': metrics,
        'matching_time': matching_time,
        'gt_match': gt_match,
        'experiment_type': graph_name_suffix
    }


def run_pgm_msd_experiment(pgm_model, data1, data2, gt_perm):
    """
    Run a matching experiment on a MSD dataset pair (already in PyG format).

    data1, data2 are PyG Data objects — predict_matching_matrix is called directly,
    bypassing infer_matching (which expects NetworkX graphs).
    Node indices are used as node IDs since MSD data has no named nodes.

    Returns:
        dict with the same keys as run_pgm_matching_experiment
    """
    start_time = time.time()
    matching_matrix = predict_matching_matrix(pgm_model.model, data1, data2, discrete=True)
    matching_time = time.time() - start_time

    rows, cols = np.where(matching_matrix.cpu().numpy() > 0)
    predicted_matches = [[str(r), str(c)] for r, c in zip(rows, cols)]

    gt_pairs = gt_perm.nonzero(as_tuple=False)
    gt_matches = [[str(r.item()), str(c.item())] for r, c in gt_pairs]

    metrics = compute_metrics(gt_matches, predicted_matches)

    return {
        'success': len(predicted_matches) > 0,
        'matches_node_ids': [predicted_matches],
        'metrics': metrics,
        'matching_time': matching_time,
        'gt_match': gt_matches,
        'experiment_type': 'pgm_msd'
    }


pickle_datasets_path = "/home/adminpc/datasets_matching/pickles"
GNN_PATH = '/root/workspace/src/graph_matching_gnn/GNN'
MSD_TEST_PATH = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise", "test_dataset.pkl")

# Load PGM model once (only needed for pgm and msd modes)
if USE_PGM:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pgm_model_instance = PartialGraphMatching(
        model_class=MatchingModel_GATv2SinkhornTopK,
        data_paths={
            "equal":   os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal"),
            "partial": os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise"),
        },
        model_save_path=os.path.join(GNN_PATH, "models", "partial_graph_matching", "ws_room_dropout_noise"),
        device=device, in_dim=7,
    )
    pgm_model_instance.load_best_model()
    print("PGM model loaded.")
else:
    pgm_model_instance = None

# Initialize data collection
all_metrics_data = []  # List of (n_nodes_s, n_nodes_a, metrics_dict)

if DATASET == "msd":
    # ── MSD dataset loop (PGM only — no classic matcher for raw PyG tensors) ──
    with open(MSD_TEST_PATH, 'rb') as f:
        msd_test_list = pickle.load(f)
    print(f"MSD test set loaded: {len(msd_test_list)} pairs")

    for data1, data2, gt_perm in tqdm(msd_test_list, desc="MSD matching", colour="green"):
        results = run_pgm_msd_experiment(pgm_model_instance, data1, data2, gt_perm)

        metrics = results['metrics'].copy() if results['metrics'] else {}
        metrics['experiment_type'] = 'pgm_msd'
        metrics['matching_time'] = results['matching_time']
        metrics['num_solutions'] = 1
        metrics['success'] = bool(results['success'])
        n_rooms_s = sum(1 for n in data2.node_names if n.endswith('_centroid'))
        n_rooms_a = sum(1 for n in data1.node_names if n.endswith('_centroid'))
        all_metrics_data.append((n_rooms_s, n_rooms_a, metrics))

    # Aggregated summary over all MSD pairs
    msd_metrics = [m for _, _, m in all_metrics_data if m.get('experiment_type') == 'pgm_msd']
    if msd_metrics:
        n = len(msd_metrics)
        avg = lambda key: np.mean([m[key] for m in msd_metrics if key in m])
        print("\n" + "="*60)
        print(f"MSD SUMMARY ({n} pairs)")
        print("="*60)
        print(f"Precision : {avg('precision'):.4f}")
        print(f"Recall    : {avg('recall'):.4f}")
        print(f"F1-Score  : {avg('f1_score'):.4f}")
        print(f"Accuracy  : {avg('accuracy'):.4f}")
        print(f"Avg Time  : {avg('matching_time'):.4f}s")
        print("="*60)

elif DATASET == "synthetic":
    # ── Synthetic dataset loop ────────────────────────────────────────────────
    full_dataset = pickle.load(open(os.path.join(pickle_datasets_path, "incremental_translation.pkl"), "rb"))

    for a_graph, s_graphs in tqdm(full_dataset, desc="Matching graphs", colour="green"):
        process_synthetic_graph(a_graph)
        for i, s_graph in enumerate(s_graphs):
            process_synthetic_graph(s_graph)
            a_graph.name = "A-Graph"
            s_graph.name = "S-Graph"

            a_graph_nodes_ids = copy.deepcopy(a_graph.get_nodes_ids())
            a_graph.stringify_node_ids()
            mapping = dict(zip(s_graph.get_nodes_ids(), list(np.array(s_graph.get_nodes_ids()) + len(a_graph_nodes_ids) + 1)))

            n_rooms_s_graphs = len(copy.deepcopy(s_graph).filter_graph_by_node_types(["room"]).get_nodes_ids())
            n_rooms_a_graphs = len(copy.deepcopy(a_graph).filter_graph_by_node_types(["room"]).get_nodes_ids())

            s_graph.stringify_node_ids()

            print(f"Running experiment with {n_rooms_s_graphs} rooms in S-graph, {n_rooms_a_graphs} in A-graph...")

            a_graph_no_objects = copy.deepcopy(a_graph).filter_graph_by_node_types(["room", "ws"])
            s_graph_no_objects = copy.deepcopy(s_graph).filter_graph_by_node_types(["room", "ws"])
            gt_match = [[i, i] for i in s_graph_no_objects.get_nodes_ids()]

            if USE_PGM:
                results_no_objects = run_pgm_matching_experiment(
                    pgm_model_instance, a_graph_no_objects, s_graph_no_objects, gt_match,
                    graph_name_suffix=f"pgm_{n_rooms_s_graphs}_rooms"
                )
            else:
                results_no_objects = run_graph_matching_experiment(
                    a_graph_no_objects, s_graph_no_objects, gt_match,
                    graph_name_suffix=f"no_objects_{n_rooms_s_graphs}_rooms"
                )

            metrics_no_objects = results_no_objects['metrics'].copy() if results_no_objects['metrics'] else {}
            metrics_no_objects['experiment_type'] = 'pgm' if USE_PGM else 'classic'
            metrics_no_objects['matching_time'] = results_no_objects['matching_time']
            metrics_no_objects['num_solutions'] = len(results_no_objects['matches_node_ids'])
            metrics_no_objects['success'] = bool(results_no_objects['success'])
            all_metrics_data.append((n_rooms_s_graphs, n_rooms_a_graphs, metrics_no_objects))

    # Aggregated summary over all synthetic pairs
    exp_label = 'pgm' if USE_PGM else 'classic'
    syn_metrics = [m for _, _, m in all_metrics_data if m.get('experiment_type') == exp_label]
    if syn_metrics:
        n = len(syn_metrics)
        avg = lambda key: np.mean([m[key] for m in syn_metrics if key in m])
        print("\n" + "="*60)
        print(f"SYNTHETIC SUMMARY ({n} pairs, matcher={exp_label})")
        print("="*60)
        print(f"Precision : {avg('precision'):.4f}")
        print(f"Recall    : {avg('recall'):.4f}")
        print(f"F1-Score  : {avg('f1_score'):.4f}")
        print(f"Accuracy  : {avg('accuracy'):.4f}")
        print(f"Avg Time  : {avg('matching_time'):.4f}s")
        print("="*60)

elif DATASET == "real":
    # ── Real environment dataset (graph_dicts/) ───────────────────────────────
    a_graph, s_graph, gt_match = load_real_graphs()

    exp_type = 'pgm_real' if USE_PGM else 'classic_real'

    if USE_PGM:
        results = run_pgm_matching_experiment(
            pgm_model_instance, a_graph, s_graph, gt_match,
            graph_name_suffix=exp_type
        )
    else:
        results = run_graph_matching_experiment(
            a_graph, s_graph, gt_match,
            graph_name_suffix=exp_type
        )

    metrics = results['metrics'].copy() if results['metrics'] else {}
    metrics['experiment_type'] = exp_type
    metrics['matching_time'] = results['matching_time']
    metrics['num_solutions'] = len(results['matches_node_ids'])
    metrics['success'] = bool(results['success'])

    n_rooms_s = sum(1 for _, attrs in s_graph.graph.nodes(data=True) if attrs.get('type') == 'room')
    n_rooms_a = sum(1 for _, attrs in a_graph.graph.nodes(data=True) if attrs.get('type') == 'room')
    all_metrics_data.append((n_rooms_s, n_rooms_a, metrics))

    print(f"\n[Real] success={metrics['success']}  solutions={metrics['num_solutions']}  "
          f"time={metrics['matching_time']:.3f}s")
    if results['metrics']:
        compute_metrics(gt_match, results['matches_node_ids'][0], log_level=1)

else:
    raise ValueError(f"Unknown DATASET value: '{DATASET}'. Choose 'synthetic', 'msd', or 'real'.")
        


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
    for n_rooms_s, n_rooms_a, metrics in all_metrics_data:
        serializable_metrics = {}
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, bool):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = float(value) if isinstance(value, np.number) else value
        
        serializable_data.append({
            'n_rooms_s_graphs': int(n_rooms_s),
            'n_rooms_a_graphs': int(n_rooms_a),
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
        'dataset_file': MSD_TEST_PATH if DATASET == "msd" else ('incremental_translation.pkl' if DATASET == "synthetic" else 'graph_dicts/'),
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