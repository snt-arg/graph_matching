
import matplotlib.pyplot as plt
import copy
import sys, json, os, re
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


# Noise condition for MSD dataset — pick one:
#   "ws_dropout_noise"        → WS noise only
#   "room_dropout_noise"      → Room noise only
#   "ws_room_dropout_noise"   → WS + Room noise (combined)
#   "ws_room_dropout_noise_inc" → WS + Room noise incremental
NOISE_CONDITION = "ws_room_dropout_noise"


# Post soft-topk threshold.
# Matches whose soft score is below this value are rejected after soft-topk.
# Drawback: a rejected node loses its match entirely (FN risk).
# Set to None to disable.
SCORE_THRESHOLD = None  # e.g. 0.3, 0.5, 0.7


# Post-Sinkhorn threshold (PGM only).
# Entries in the doubly-stochastic matrix S below this value are zeroed before
# Hungarian, guiding the solver away from weak pairs. S values are in [0, 1]
# so the threshold should be in that range (e.g. 0.1, 0.2).
# Set to None to disable (Hungarian runs on the full S matrix).
SINKHORN_THRESHOLD = None # e.g. 0.1, 0.2 — None disables


# Pre-normalisation accuracy threshold (PGM only).
# Entries in the raw dot-product similarity matrix (before instance norm) below
# this value are masked to -inf in sim_normed before Sinkhorn, so the solver
# never distributes mass to clearly implausible pairs.
# The threshold is in the raw embedding dot-product space (not [0,1]).
# Set to None to disable.
ACC_THRESHOLD = None # e.g. -1.0, 0.0, 1.0 — None disables


# Monte Carlo Dropout samples (PGM only).
# When > 0, runs MC Dropout inference with this many stochastic forward passes
# and averages the soft matrices before Hungarian. Also stores per-entry
# uncertainty (std across passes) in experiment results.
# Set to 0 to disable (standard single-pass inference).
MC_SAMPLES = 0 # e.g. 10, 30, 50


# MC Dropout std threshold (PGM only, requires MC_SAMPLES > 0).
# Entries in mean_soft whose uncertainty (std across passes) exceeds this value
# are zeroed before Hungarian, filtering candidates the model was inconsistent about.
# Set to None to disable.
# STD_THRESHOLD = 0.417  # best F1 threshold found from plot_results.py --mc-dropout (with mc_samples=30)
# STD_THRESHOLD = None  # best F1 threshold found from plot_results.py --mc-dropout
# STD_THRESHOLD = 0.442  # best F1 threshold found from plot_results.py --mc-dropout (with mc_samples=10)



# Per-experiment timeout for the classic matcher (seconds).
# Pairs that exceed this limit are skipped and counted separately.
# Used to demonstrate that the classic matcher lacks deterministic runtime.
# Set to None to disable (classic matcher may hang on large graphs).
CLASSIC_TIMEOUT_S = 60  # e.g. 10, 30, 60


if USE_PGM:
    import torch
    PGM_PATH = '/root/workspace/src/graph_matching_gnn/graph_matching'
    if PGM_PATH not in sys.path:
        sys.path.insert(0, PGM_PATH)
    from PGM_class import PartialGraphMatching, MatchingModel_GATv2SinkhornTopK, predict_matching_matrix  # type: ignore
else:
    from GraphMatcher import GraphMatcher
    if DATASET == "msd":
        import torch  # needed to unpack PyG-formatted MSD test data even without the GNN model



# reasoning_msgs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# with open(os.path.join(reasoning_msgs,"config", "syntheticDS_params_synthetic.json")) as f:
#     syntheticDS_params = json.load(f)


syntheticDS_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "syntheticDS_params_synthetic.json")
with open(syntheticDS_params_path) as f:
    syntheticDS_params = json.load(f)


realDS_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "syntheticDS_params_real.json")
with open(realDS_params_path) as f:
    realDS_params = json.load(f)


synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
# from situational_graphs_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
# from situational_graphs_datasets.graph_visualizer import visualize_nxgraph_3d


import networkx as nx
import situational_graphs_wrapper
from situational_graphs_wrapper.GraphWrapper import GraphWrapper
# Make the module available under both names for pickle compatibility
sys.modules['graph_wrapper'] = situational_graphs_wrapper


# from situational_graphs_datasets.config import get_config as get_datasets_config
# synteticdataset_settings = get_datasets_config("graph_matching")


def pyg_data_to_graphwrapper(data, original_graphs, name=None):
    """Reconstruct a GraphWrapper from a PyG Data object by looking up the
    original NetworkX DiGraph (which has full geometry: center, normal, limits).


    data.name is the graph index/name used to match against original_graphs.
    original_graphs is the list of NX DiGraphs from original.pkl.
    """
    orig_nx = next((g for g in original_graphs if str(g.graph.get('name')) == str(data.name)), None)
    if orig_nx is None:
        raise ValueError(f"No original graph found with name '{data.name}'")


    # Only keep the nodes present in data (the partial/noisy subset)
    G = nx.DiGraph()
    display_name = name if name is not None else str(data.name)
    G.graph['name'] = display_name
    perm = data.permutation.tolist()
    node_ids = [data.node_names[idx] for idx in perm]
    for node_id in node_ids:
        if node_id in orig_nx.nodes:
            G.add_node(node_id, **orig_nx.nodes[node_id])
    for u_idx, v_idx in data.edge_index.t().tolist():
        u = node_ids[u_idx]
        v = node_ids[v_idx]
        if u in G and v in G:
            edge_attrs = orig_nx.edges[u, v] if orig_nx.has_edge(u, v) else {}
            G.add_edge(u, v, **edge_attrs)


    wrapper = GraphWrapper(graph_obj=G)
    wrapper.name = display_name
    return wrapper



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
            c = node_attrs["center"]
            node_attrs["Geometric_info"] = np.array([c[0], c[1], c[2] if len(c) > 2 else 0.0])
        elif node_attrs["type"] == "ws":
            c = node_attrs["center"]
            n = node_attrs["normal"]
            initial = [c[0], c[1], c[2] if len(c) > 2 else 0.0,
                       n[0], n[1], n[2] if len(n) > 2 else 0.0]


            # node_attrs["Geometric_info"] = plane_4_params_to_6_params(plane_6_params_to_4_params(initial))
            node_attrs["Geometric_info"] = initial



def unsplit_graph(graph):
    """Merge GNN split-plane nodes (e.g. '92077532_s0') back to their original ws node.


    GNN-format graphs split wall planes into sub-segments for GNN processing.
    The classic matcher works on whole walls, so split nodes must be merged back.
    Each split node has an 'original_id' pointing to the original plane ID and
    'original_attrs' carrying the full-wall center/normal.
    """
    g = graph.graph


    splits_by_original = {}
    for node_id, attrs in list(g.nodes(data=True)):
        original_id = attrs.get("original_id")
        if original_id is not None:
            splits_by_original.setdefault(original_id, []).append((node_id, attrs))


    for original_id, split_nodes in splits_by_original.items():
        first_attrs = split_nodes[0][1]
        orig_attrs = first_attrs.get("original_attrs", {})


        center = orig_attrs.get("center", first_attrs.get("center", [0., 0.]))
        normal = orig_attrs.get("normal", first_attrs.get("normal", [0., 0.]))
        if hasattr(center, "tolist"):
            center = center.tolist()
        if hasattr(normal, "tolist"):
            normal = normal.tolist()


        split_ids = {s[0] for s in split_nodes}
        all_neighbors = set()
        for split_id in split_ids:
            if hasattr(g, 'predecessors'):
                all_neighbors.update(g.predecessors(split_id))
                all_neighbors.update(g.successors(split_id))
            else:
                all_neighbors.update(g.neighbors(split_id))
        all_neighbors -= split_ids


        g.add_node(original_id,
                   type="ws",
                   center=center,
                   normal=normal,
                   original_type=first_attrs.get("original_type", "Plane"),
                   original_attrs=orig_attrs)


        for neighbor in all_neighbors:
            g.add_edge(neighbor, original_id)


        for split_id in split_ids:
            g.remove_node(split_id)



def strip_split_suffix(node_id):
    """Convert a split node ID like '92077532_s4' to its original ID '92077532'."""
    return re.sub(r'_s\d+$', '', str(node_id))



def map_gt_to_unsplit_ids(gt_match):
    """Remap GT pairs from split-node IDs to original unsplit node IDs.


    The ground truth JSON was created from GNN-format (split) graphs, so ws IDs
    look like '92077532_s4'. After unsplit_graph(), those nodes no longer exist —
    only the original '92077532' remains. Strip the '_sX' suffix so the GT IDs
    align with the unsplit graph. Room IDs (plain integers) are unaffected.
    """
    seen = set()
    mapped = []
    for prior_id, online_id in gt_match:
        pair = (strip_split_suffix(prior_id), strip_split_suffix(online_id))
        if pair not in seen:
            seen.add(pair)
            mapped.append(list(pair))
    return mapped



def process_real_graph(graph):
    """Set Geometric_info on real-environment graph nodes.


    Call after unsplit_graph(). The classic matcher requires Geometric_info in
    the same representation used by the node's graph_callback:
      - room: [cx, cy, cz]  (room center)
      - ws:   [cx, cy, cz, nx, ny, nz]  where [cx,cy,cz] = closest point on
              the infinite plane to the world origin (NOT the segment center).


    The correct Geometric_info is preserved in original_attrs on each node
    (stored by convert_wrapper_to_gnn_format before GNN splitting). Using the
    segment center stored in node_attrs["center"] gives the wrong invariants and
    causes the matcher to fail on real environments.
    """
    graph.to_undirected()
    for node_id, node_attrs in graph.get_attributes_of_all_nodes():
        original_attrs = node_attrs.get("original_attrs", {})
        if node_attrs.get("type") == "room":
            center = node_attrs.get("center", [0., 0.])
            node_attrs["Geometric_info"] = np.array([center[0], center[1], 0.0])
        elif node_attrs.get("type") == "ws":
            orig_geom = original_attrs.get("Geometric_info")
            if orig_geom is not None:
                orig_geom = np.array(orig_geom, dtype=float)
                if len(orig_geom) >= 6:
                    # Use canonical x,y (closest-point-to-origin), force z=0.
                    # Prior maps are 2D (z=0 already); Online SLAM estimates may
                    # have small non-zero z/nz from 3D SLAM.  Forcing z=0 for
                    # both keeps the invariants consistent, matching what the node
                    # does when it reconstructs Geometric_info in graph_callback.
                    node_attrs["Geometric_info"] = np.array([
                        orig_geom[0], orig_geom[1], 0.0,
                        orig_geom[3], orig_geom[4], 0.0,
                    ])
                else:
                    center = node_attrs.get("center", [0., 0.])
                    normal = node_attrs.get("normal", [0., 0.])
                    node_attrs["Geometric_info"] = np.array([center[0], center[1], 0.0,
                                                             normal[0], normal[1], 0.0])
            else:
                center = node_attrs.get("center", [0., 0.])
                normal = node_attrs.get("normal", [0., 0.])
                node_attrs["Geometric_info"] = np.array([center[0], center[1], 0.0,
                                                         normal[0], normal[1], 0.0])



def process_node_saved_graph(graph):
    """Prepare a GraphWrapper saved directly by the ROS node for the classic matcher.


    The node stores graphs with "Finite Room" and "Plane" node types and already
    populates Geometric_info correctly (6-param planes, room center).  The matcher
    expects "room" and "ws" type names, so we remap them here.  No unsplitting or
    Geometric_info recomputation is needed.
    """
    graph.to_undirected()
    for node_id, node_attrs in graph.get_attributes_of_all_nodes():
        if node_attrs.get("type") == "Finite Room":
            node_attrs["type"] = "room"
            gi = np.array(node_attrs["Geometric_info"], dtype=float)
            # ensure 3D point [cx, cy, cz]
            if len(gi) == 2:
                node_attrs["Geometric_info"] = np.array([gi[0], gi[1], 0.0])
            else:
                node_attrs["Geometric_info"] = np.array([gi[0], gi[1], float(gi[2]) if len(gi) > 2 else 0.0])
        elif node_attrs.get("type") == "Plane":
            node_attrs["type"] = "ws"
            gi = np.array(node_attrs["Geometric_info"], dtype=float)
            # node already converts 4-param → 6-param; just force z/nz = 0 for 2D consistency
            if len(gi) >= 6:
                node_attrs["Geometric_info"] = np.array([gi[0], gi[1], 0.0,
                                                         gi[3], gi[4], 0.0])



REAL_ENVS = ["47_basement_Unsplitted", "47_topfloor_", "CF12"]


def load_real_graphs(env_name):
    """Load Prior/Online graphs and ground truth from graph_dicts/<env_name>/.


    The pickled graphs are already in GNN format (split planes, sanitized attrs).
    Ground truth JSON format:
        { "rooms": {"online_id": "prior_id", ...},
          "ws":    [["online_id", "prior_id"], ...] }


    Returns:
        (a_graph, s_graph, gt_match)
        where gt_match = [[prior_node_id, online_node_id], ...] (A-graph, S-graph order)
        gt_match is [] when no ground_truth.json exists for this environment.
    """
    graph_dicts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_dicts", env_name)


    def _load_graph_pkl(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Node-saved pkls contain a raw NX DiGraph; GNN pkls contain a GraphWrapper.
        if isinstance(obj, GraphWrapper):
            return obj
        return GraphWrapper(graph_obj=obj)


    a_graph = _load_graph_pkl(os.path.join(graph_dicts_path, "Prior.pkl"))
    s_graph = _load_graph_pkl(os.path.join(graph_dicts_path, "Online.pkl"))


    # Sanitize numpy arrays so the PGM feature builder (list concatenation) works correctly
    for g in (a_graph, s_graph):
        for _, attrs in g.graph.nodes(data=True):
            for key in ("center", "normal"):
                if key in attrs and hasattr(attrs[key], "tolist"):
                    attrs[key] = attrs[key].tolist()


    # Load and parse ground truth — store as [[prior_id, online_id]] = [[A, S]]
    gt_match = []
    gt_path = os.path.join(graph_dicts_path, "ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            raw = json.load(f)
        for o_id, p_id in raw.get("rooms", {}).items():
            if p_id != "??":
                gt_match.append([str(p_id), str(o_id)])   # [prior=A, online=S]
        for entry in raw.get("ws", []):
            if len(entry) == 2 and entry[1] != "??":
                gt_match.append([str(entry[1]), str(entry[0])])  # [prior=A, online=S]


    gt_info = f"GT pairs: {len(gt_match)}" if gt_match else "no ground truth"
    print(f"[{env_name}] Prior: {a_graph.graph.number_of_nodes()} nodes, "
          f"Online: {s_graph.graph.number_of_nodes()} nodes, {gt_info}")
    return a_graph, s_graph, gt_match
            


def run_graph_matching_experiment(a_graph, s_graph, gt_match, graph_name_suffix="", params=None):
    """
    Run a single graph matching experiment and return results. (CLASSIC MATCHER)


    Args:
        a_graph: The reference (A) graph
        s_graph: The target (S) graph
        graph_name_suffix: Suffix to add to graph names for identification
        params: Matcher parameter dict; defaults to syntheticDS_params when None.
                Pass realDS_params for real-environment graphs.


    Returns:
        dict: Dictionary containing success status, metrics, timing, and other results
    """
    if params is None:
        params = syntheticDS_params
    # Create GraphMatcher
    graph_matcher = GraphMatcher(fake_logger, log_level=0)
    graph_matcher.set_parameters(params)
    graph_matcher.set_graph_from_wrapper(a_graph, "A-Graph")
    graph_matcher.set_graph_from_wrapper(s_graph, "S-Graph")
    
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



def run_pgm_matching_experiment(pgm_model, a_graph, s_graph, gt_match, graph_name_suffix="", sinkhorn_threshold=None, score_threshold=None, acc_threshold=None, mc_samples=0, std_threshold=None):
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
        mc_samples       : if > 0, use Monte Carlo Dropout with this many passes


    Returns:
        dict with the same keys as run_graph_matching_experiment
    """
    # infer_matching expects a raw NetworkX graph — unwrap if a GraphWrapper is passed
    g1 = a_graph.graph if hasattr(a_graph, 'graph') else a_graph
    g2 = s_graph.graph if hasattr(s_graph, 'graph') else s_graph


    start_time = time.time()
    output = pgm_model.infer_matching(g1, g2, discrete=True, sinkhorn_threshold=sinkhorn_threshold, score_threshold=score_threshold, acc_threshold=acc_threshold, mc_samples=mc_samples, std_threshold=std_threshold)
    matching_time = time.time() - start_time
    matching_matrix, uncertainty = output if mc_samples > 0 else (output, None)


    g1_nodes = list(g1.nodes())
    g2_nodes = list(g2.nodes())


    rows, cols = np.where(matching_matrix.cpu().numpy() > 0)
    predicted_matches = [[str(g1_nodes[r]), str(g2_nodes[c])] for r, c in zip(rows, cols)]


    success = len(predicted_matches) > 0
    matches_node_ids = [predicted_matches]


    metrics = None
    if success and len(matches_node_ids) == 1:
        metrics = compute_metrics(gt_match, predicted_matches)


    return {
        'success': success,
        'matches_node_ids': matches_node_ids,
        'metrics': metrics,
        'matching_time': matching_time,
        'gt_match': gt_match,
        'experiment_type': graph_name_suffix,
        'uncertainty': uncertainty,
    }



def run_pgm_msd_experiment(pgm_model, data1, data2, gt_perm, sinkhorn_threshold=None, score_threshold=None, acc_threshold=None, mc_samples=0, std_threshold=None):
    """
    Run a matching experiment on a MSD dataset pair (already in PyG format).


    data1, data2 are PyG Data objects — predict_matching_matrix is called directly,
    bypassing infer_matching (which expects NetworkX graphs).
    Node indices are used as node IDs since MSD data has no named nodes.


    Returns:
        dict with the same keys as run_pgm_matching_experiment, plus 'soft_S' and 'gt_perm'
    """
    start_time = time.time()
    output = predict_matching_matrix(pgm_model.model, data1, data2, discrete=True, sinkhorn_threshold=sinkhorn_threshold, score_threshold=score_threshold, acc_threshold=acc_threshold, mc_samples=mc_samples, std_threshold=std_threshold)
    matching_time = time.time() - start_time
    matching_matrix, uncertainty = output if mc_samples > 0 else (output, None)


    # Also get the soft Sinkhorn matrix (before Hungarian) for score distribution analysis
    soft_S = predict_matching_matrix(pgm_model.model, data1, data2, discrete=False, sinkhorn_threshold=None, score_threshold=None, acc_threshold=acc_threshold)


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
        'experiment_type': 'pgm_msd',
        'soft_S': soft_S.cpu().numpy(),
        'gt_perm': gt_perm.cpu().numpy(),
        'uncertainty': uncertainty.cpu().numpy() if uncertainty is not None else None,
        'matching_matrix': matching_matrix.cpu().numpy(),
    }



def run_classic_msd_experiment(data1, data2, gt_perm, original_graphs, graph_name_suffix=""):
    """Run a single MSD experiment using the classic CLIPPER-based matcher.


    Reconstructs GraphWrapper objects from PyG Data via pyg_data_to_graphwrapper,
    sets Geometric_info with process_synthetic_graph, builds ground-truth pairs from
    gt_perm, then delegates to run_graph_matching_experiment.


    gt_perm[i, j] == 1 means data1 node i corresponds to data2 node j (row/col indices
    follow the permutation order stored in data.permutation).
    """
    a_graph = pyg_data_to_graphwrapper(data1, original_graphs, name="A-Graph")
    s_graph = pyg_data_to_graphwrapper(data2, original_graphs, name="S-Graph")


    process_synthetic_graph(a_graph)
    process_synthetic_graph(s_graph)


    # GraphMatcher calls int() on every node ID, so IDs must be integer-parseable strings.
    # Relabel to 0-based ints, then stringify: "0", "1", ... satisfy int("0") while
    # avoiding the np.int64 equality-returns-array bug in find_nodes_by_attrs.
    a_node_ids = [data1.node_names[i] for i in data1.permutation.tolist()]
    s_node_ids = [data2.node_names[i] for i in data2.permutation.tolist()]
    a_mapping = a_graph.relabel_nodes()   # {orig_id: int_id}
    s_mapping = s_graph.relabel_nodes()
    a_graph.stringify_node_ids()          # int_id → "0", "1", ...
    s_graph.stringify_node_ids()


    gt_pairs = gt_perm.nonzero(as_tuple=False)
    gt_matches = [[str(a_mapping[a_node_ids[r.item()]]), str(s_mapping[s_node_ids[c.item()]])]
                  for r, c in gt_pairs]


    return run_graph_matching_experiment(a_graph, s_graph, gt_matches,
                                         graph_name_suffix=graph_name_suffix)



pickle_datasets_path = "/home/adminpc/datasets_matching/pickles"
GNN_PATH = '/root/workspace/src/graph_matching_gnn/GNN'
MSD_TEST_PATH = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", NOISE_CONDITION, "test_dataset.pkl")
MSD_ORIGINAL_PATH = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal", "original.pkl")


# Load PGM model once (only needed for pgm and msd modes)
if USE_PGM:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pgm_model_instance = PartialGraphMatching(
        model_class=MatchingModel_GATv2SinkhornTopK,
        data_paths={
            "equal":   os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal"),
            "partial": os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", NOISE_CONDITION),
        },
        model_save_path=os.path.join(GNN_PATH, "models", "partial_graph_matching", NOISE_CONDITION),
        device=device, in_dim=7,
    )
    pgm_model_instance.load_best_model()
    print("PGM model loaded.")
else:
    pgm_model_instance = None


# Initialize data collection
all_metrics_data = []  # List of (n_nodes_s, n_nodes_a, metrics_dict)


if DATASET == "msd":
    with open(MSD_TEST_PATH, 'rb') as f:
        msd_test_list = pickle.load(f)
    print(f"MSD test set loaded: {len(msd_test_list)} pairs")


    if USE_PGM:
        # ── PGM matcher path ──────────────────────────────────────────────────
        # Reuse original graphs already loaded by the PGM model (avoids double memory cost)
        original_graphs_raw = pgm_model_instance.original_graphs


        # ── Visualize first 20 full original graphs from the MSD test set ─────
        # for idx in range(min(20, len(msd_test_list))):
        #     data1, _, _ = msd_test_list[idx]
        #     orig_nx = next((g for g in original_graphs_raw if str(g.graph.get('name')) == str(data1.name)), None)
        #     if orig_nx is None:
        #         print(f"[WARN] No original graph found for MSD pair {idx} (name={data1.name})")
        #         continue
        #     g_viz = GraphWrapper(graph_obj=copy.deepcopy(orig_nx))
        #     g_viz.name = f"MSD_{idx}_orig"
        #     g_viz.from_2D_to_3D()
        #     g_viz._add_complete_viz_attributes_to_graph()
        #     visualize_nxgraph_3d(g_viz, f"MSD_{idx}_orig", visualize_alone=True,
        #                          include_node_ids=False, blocking=False)
        # plt.show(block=True)
        # ─────────────────────────────────────────────────────────────────────


        all_tp_scores = []
        all_fp_scores = []
        all_hard_tp_scores = []
        all_hard_fp_scores = []
        all_assigned_tp_uncertainty = []  # uncertainty at Hungarian-assigned correct matches
        all_assigned_fp_uncertainty = []  # uncertainty at Hungarian-assigned wrong matches
        total_gt_matches = 0              # total GT matches across test set (TP + FN)


        for data1, data2, gt_perm in tqdm(msd_test_list, desc="MSD matching", colour="green"):
            results = run_pgm_msd_experiment(pgm_model_instance, data1, data2, gt_perm, sinkhorn_threshold=SINKHORN_THRESHOLD, score_threshold=SCORE_THRESHOLD, acc_threshold=ACC_THRESHOLD, mc_samples=MC_SAMPLES, std_threshold=STD_THRESHOLD)


            metrics = results['metrics'].copy() if results['metrics'] else {}
            metrics['experiment_type'] = 'pgm_msd'
            metrics['matching_time'] = results['matching_time']
            metrics['num_solutions'] = 1
            metrics['success'] = bool(results['success'])
            n_rooms_s = sum(1 for n in data2.node_names if n.endswith('_centroid'))
            n_rooms_a = sum(1 for n in data1.node_names if n.endswith('_centroid'))
            all_metrics_data.append((n_rooms_s, n_rooms_a, metrics))


            # Collect Sinkhorn TP/FP scores for distribution plot
            soft_S = results['soft_S']   # [N1, N2]
            gt_mask = results['gt_perm'] # [N1, N2], 1=TP, 0=FP
            h, w = soft_S.shape
            gt_mask_crop = gt_mask[:h, :w]
            all_tp_scores.append(soft_S[gt_mask_crop == 1].flatten())
            all_fp_scores.append(soft_S[gt_mask_crop == 0].flatten())

            # Hard-assignment TP/FP: only the cells Hungarian selected
            hard = results['matching_matrix']  # [N1, N2] binary
            hard_crop = hard[:h, :w]
            all_hard_tp_scores.append(soft_S[(hard_crop == 1) & (gt_mask_crop == 1)].flatten())
            all_hard_fp_scores.append(soft_S[(hard_crop == 1) & (gt_mask_crop == 0)].flatten())


            # Collect MC Dropout uncertainty at Hungarian-assigned positions only
            if results['uncertainty'] is not None:
                unc  = results['uncertainty']    # [N1, N2]
                hard = results['matching_matrix'] # [N1, N2] binary
                hu, wu = unc.shape
                gt_mask_unc  = gt_mask[:hu, :wu]
                hard_crop    = hard[:hu, :wu]
                total_gt_matches += int((gt_mask_unc == 1).sum())
                # Assigned TP: Hungarian selected AND gt match
                all_assigned_tp_uncertainty.append(unc[(hard_crop == 1) & (gt_mask_unc == 1)].flatten())
                # Assigned FP: Hungarian selected AND not gt match
                all_assigned_fp_uncertainty.append(unc[(hard_crop == 1) & (gt_mask_unc == 0)].flatten())


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


        # Sinkhorn score distribution plot (TP vs FP)
        if all_tp_scores or all_fp_scores:
            from scipy.stats import gaussian_kde


            tp_scores = np.concatenate(all_tp_scores) if all_tp_scores else np.array([])
            fp_scores = np.concatenate(all_fp_scores) if all_fp_scores else np.array([])


            fig, ax = plt.subplots(figsize=(8, 5))
            x_range = np.linspace(0, 1, 500)
            colors = {'tp': '#2ca02c', 'fp': '#d62728'}
            for label, arr, color in [
                (f'GT match / TP  (n={len(tp_scores):,})',     tp_scores, colors['tp']),
                (f'GT non-match / FP  (n={len(fp_scores):,})', fp_scores, colors['fp']),
            ]:
                if len(arr) > 1:
                    kde = gaussian_kde(arr)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                    ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Sinkhorn score')
            ax.set_ylabel('Density')
            ax.set_title('Sinkhorn score distribution — TP vs FP (MSD test set)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()


            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, "sinkhorn_score_distribution.png")
            fig.savefig(plot_path, dpi=150)
            print(f"Sinkhorn score distribution plot saved to: {plot_path}")
            plt.show()

        # Hard-assignment score distribution plot (TP vs FP on the permutation matrix)
        hard_tp = np.clip(np.concatenate(all_hard_tp_scores), 0.0, 1.0) if all_hard_tp_scores and any(len(a) for a in all_hard_tp_scores) else np.array([])
        hard_fp = np.clip(np.concatenate(all_hard_fp_scores), 0.0, 1.0) if all_hard_fp_scores and any(len(a) for a in all_hard_fp_scores) else np.array([])
        if len(hard_tp) > 1 or len(hard_fp) > 1:
            from scipy.stats import gaussian_kde

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            x_range = np.linspace(0, 1, 500)
            colors = {'tp': '#2ca02c', 'fp': '#d62728'}
            for label, arr, color in [
                (f'TP — correct assignments  (n={len(hard_tp):,})', hard_tp, colors['tp']),
                (f'FP — wrong assignments    (n={len(hard_fp):,})', hard_fp, colors['fp']),
            ]:
                if len(arr) > 1:
                    kde = gaussian_kde(arr)
                    ax2.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                    ax2.fill_between(x_range, kde(x_range), alpha=0.15, color=color)
                elif len(arr) == 1:
                    ax2.axvline(arr[0], color=color, linewidth=2, linestyle='--', label=f'{label} (single point)')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 20)
            ax2.set_xlabel('Sinkhorn score')
            ax2.set_ylabel('Density')
            ax2.set_title('Score distribution on Hungarian assignments — TP vs FP (MSD test set)')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()

            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(results_dir, exist_ok=True)
            hard_plot_path = os.path.join(results_dir, "hard_assignment_score_distribution.png")
            fig2.savefig(hard_plot_path, dpi=150)
            print(f"Hard-assignment score distribution plot saved to: {hard_plot_path}")
            plt.show()


    else:
        # ── Classic matcher path ──────────────────────────────────────────────
        # original.pkl holds full-geometry NX graphs used to reconstruct GraphWrappers
        with open(MSD_ORIGINAL_PATH, 'rb') as f:
            original_graphs_raw = pickle.load(f)
        print(f"MSD original graphs loaded: {len(original_graphs_raw)}")


        import signal


        def _timeout_handler(signum, frame):
            raise TimeoutError("classic matcher exceeded time limit")


        n_timed_out = 0
        total_pairs = len(msd_test_list)


        for data1, data2, gt_perm in tqdm(msd_test_list, desc="MSD classic matching", colour="green"):
            n_rooms_s = sum(1 for n in data2.node_names if n.endswith('_centroid'))
            n_rooms_a = sum(1 for n in data1.node_names if n.endswith('_centroid'))


            if CLASSIC_TIMEOUT_S is not None:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(CLASSIC_TIMEOUT_S)
            try:
                results = run_classic_msd_experiment(data1, data2, gt_perm, original_graphs_raw,
                                                     graph_name_suffix="classic_msd")
                if CLASSIC_TIMEOUT_S is not None:
                    signal.alarm(0)  # cancel alarm on success
            except TimeoutError:
                if CLASSIC_TIMEOUT_S is not None:
                    signal.alarm(0)
                n_timed_out += 1
                all_metrics_data.append((n_rooms_s, n_rooms_a, {
                    'experiment_type': 'classic_msd',
                    'timed_out': True,
                    'matching_time': CLASSIC_TIMEOUT_S,
                    'success': False,
                    'num_solutions': 0,
                }))
                continue


            metrics = results['metrics'].copy() if results['metrics'] else {}
            metrics['experiment_type'] = 'classic_msd'
            metrics['timed_out'] = False
            metrics['matching_time'] = results['matching_time']
            metrics['num_solutions'] = len(results['matches_node_ids'])
            metrics['success'] = bool(results['success'])
            all_metrics_data.append((n_rooms_s, n_rooms_a, metrics))


        # Aggregated summary over all MSD pairs
        classic_msd_metrics = [m for _, _, m in all_metrics_data if m.get('experiment_type') == 'classic_msd']
        completed_metrics = [m for m in classic_msd_metrics if not m.get('timed_out', False)]
        if classic_msd_metrics:
            n_completed = len(completed_metrics)
            avg = lambda key: np.mean([m[key] for m in completed_metrics if key in m])
            std = lambda key: np.std([m[key] for m in completed_metrics if key in m])
            timeout_pct = 100.0 * n_timed_out / total_pairs
            print("\n" + "="*60)
            print(f"MSD CLASSIC SUMMARY  (timeout={CLASSIC_TIMEOUT_S}s)")
            print("="*60)
            print(f"Total pairs          : {total_pairs}")
            print(f"Completed            : {n_completed}  ({100-timeout_pct:.1f}%)")
            print(f"Timed out (skipped)  : {n_timed_out}  ({timeout_pct:.1f}%)")
            print("─"*60)
            if n_completed > 0:
                print(f"Precision : {avg('precision'):.4f}")
                print(f"Recall    : {avg('recall'):.4f}")
                print(f"F1-Score  : {avg('f1_score'):.4f}")
                print(f"Accuracy  : {avg('accuracy'):.4f}")
                print(f"Avg Time  : {avg('matching_time'):.4f}s  (std: {std('matching_time'):.4f}s)")
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
            gt_match = [[node_id, node_id] for node_id in s_graph_no_objects.get_nodes_ids()]


            if USE_PGM:
                results_no_objects = run_pgm_matching_experiment(
                    pgm_model_instance, a_graph_no_objects, s_graph_no_objects, gt_match,
                    graph_name_suffix=f"pgm_{n_rooms_s_graphs}_rooms",
                    sinkhorn_threshold=SINKHORN_THRESHOLD, score_threshold=SCORE_THRESHOLD, acc_threshold=ACC_THRESHOLD, mc_samples=MC_SAMPLES, std_threshold=STD_THRESHOLD
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
    # ── Real environment dataset (graph_dicts/<env>/) ─────────────────────────
    exp_type = 'pgm_real' if USE_PGM else 'classic_real'
    summary_rows = []  # collect per-env results for the summary table


    for env_name in REAL_ENVS:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_name}")
        print(f"{'='*60}")


        a_graph, s_graph, gt_match = load_real_graphs(env_name)
        already_unsplit = env_name.endswith("_Unsplitted")
        if not USE_PGM:
            if already_unsplit:
                # graphs saved directly by the node: Finite Room/Plane types, Geometric_info
                # already set — only remap type names for the matcher
                process_node_saved_graph(a_graph)
                process_node_saved_graph(s_graph)
                # GT IDs have no _sX suffixes, so no remapping needed
            else:
                unsplit_graph(a_graph)
                unsplit_graph(s_graph)
                process_real_graph(a_graph)
                process_real_graph(s_graph)
                gt_match = map_gt_to_unsplit_ids(gt_match)


        if USE_PGM:
            results = run_pgm_matching_experiment(
                pgm_model_instance, a_graph, s_graph, gt_match,
                graph_name_suffix=f"{exp_type}_{env_name}",
                sinkhorn_threshold=SINKHORN_THRESHOLD, score_threshold=SCORE_THRESHOLD, acc_threshold=ACC_THRESHOLD, mc_samples=MC_SAMPLES, std_threshold=STD_THRESHOLD
            )
        else:
            results = run_graph_matching_experiment(
                a_graph, s_graph, gt_match,
                graph_name_suffix=f"{exp_type}_{env_name}",
                params=realDS_params
            )


        metrics = results['metrics'].copy() if results['metrics'] else {}
        metrics['experiment_type'] = f"{exp_type}_{env_name}"
        metrics['matching_time'] = results['matching_time']
        metrics['num_solutions'] = len(results['matches_node_ids'])
        metrics['success'] = bool(results['success'])
        metrics['environment'] = env_name


        n_rooms_s = sum(1 for _, attrs in s_graph.graph.nodes(data=True) if attrs.get('type') == 'room')
        n_rooms_a = sum(1 for _, attrs in a_graph.graph.nodes(data=True) if attrs.get('type') == 'room')
        all_metrics_data.append((n_rooms_s, n_rooms_a, metrics))


        if gt_match and results['metrics']:
            compute_metrics(gt_match, results['matches_node_ids'][0], log_level=1)


        summary_rows.append({
            'env': env_name,
            'precision': metrics.get('precision', float('nan')),
            'recall':    metrics.get('recall',    float('nan')),
            'f1':        metrics.get('f1_score',   float('nan')),
            'accuracy':  metrics.get('accuracy',  float('nan')),
            'time':      metrics['matching_time'],
            'has_gt':    bool(gt_match),
        })


    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  REAL ENVIRONMENT SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Environment':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Time(s)':>9}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*9}")
    for row in summary_rows:
        if row['has_gt']:
            print(f"  {row['env']:<22} {row['precision']:>10.4f} {row['recall']:>8.4f} "
                  f"{row['f1']:>8.4f} {row['accuracy']:>10.4f} {row['time']:>9.3f}")
        else:
            print(f"  {row['env']:<22} {'n/a':>10} {'n/a':>8} {'n/a':>8} {'n/a':>10} {row['time']:>9.3f}")
    print(f"{'='*72}")


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
        'description': {
            'msd':       'Graph matching performance metrics collected from MSD dataset experiments',
            'synthetic': 'Graph matching performance metrics collected from synthetic dataset experiments',
            'real':      'Graph matching performance metrics collected from real environment dataset experiments',
        }.get(DATASET, f'Graph matching performance metrics collected from {DATASET} dataset experiments'),
        'score_threshold': SCORE_THRESHOLD,
        'sinkhorn_threshold': SINKHORN_THRESHOLD,
        'acc_threshold': ACC_THRESHOLD,
        'mc_samples': MC_SAMPLES,
        'std_threshold': STD_THRESHOLD,
        'dataset': DATASET,
        'noise_condition': NOISE_CONDITION,
    }


    # Aggregate MC Dropout uncertainty arrays (only present when MC_SAMPLES > 0 and DATASET == "msd")
    mc_uncertainty_data = {}
    if 'all_assigned_tp_uncertainty' in dir() and all_assigned_tp_uncertainty:
        mc_uncertainty_data['assigned_tp_uncertainty'] = np.concatenate(all_assigned_tp_uncertainty).tolist()
        mc_uncertainty_data['assigned_fp_uncertainty'] = np.concatenate(all_assigned_fp_uncertainty).tolist()
        mc_uncertainty_data['total_gt_matches'] = int(total_gt_matches)
        mc_uncertainty_data['mc_samples'] = MC_SAMPLES


    # Combine data and metadata
    output_data = {
        'metadata': metadata,
        'experiments': serializable_data,
        'mc_uncertainty': mc_uncertainty_data,
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