# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import rclpy, os, sys
import time
import copy
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import subprocess ## To run PGM in subprocess
import torch
import pickle   
import tempfile





from rclpy.node import Node
from .utils import *
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros.buffer_interface import BufferInterface
import ament_index_python
from ament_index_python.packages import get_package_share_directory
import tf2_geometry_msgs 
from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import PointStamped as PointStampedMsg
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Transform as TransformMsg
from geometry_msgs.msg import TransformStamped as TransformStampedMsg
from std_msgs.msg import ColorRGBA as ColorRGBSMsg
from std_msgs.msg import Header as HeaderMsg
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.parameter import Parameter
from rclpy.parameter import ParameterType





from situational_graphs_reasoning_msgs.srv import SubgraphMatch as SubgraphMatchSrv
from situational_graphs_reasoning_msgs.msg import Graph as GraphMsg
from situational_graphs_reasoning_msgs.msg import Match as MatchMsg
from situational_graphs_reasoning_msgs.msg import Node as NodeMsg
from situational_graphs_reasoning_msgs.msg import Edge as EdgeMsg
from situational_graphs_reasoning_msgs.msg import Attribute as AttributeMsg
from situational_graphs_wrapper.GraphWrapper import GraphWrapper
from situational_graphs_msgs.msg import PlanesData as PlanesDataMsg
from situational_graphs_reasoning.situational_graphs_reasoning_node import SituationalGraphReasoningNode

from situational_graphs_datasets.graph_visualizer import visualize_nxgraph_3d




from .GraphMatcher import GraphMatcher
from .utils import plane_4_params_to_6_params






################################### GNN matching





class GraphMatchingNode(Node):





    def __init__(self):
        #Classic matcher node init
        super().__init__('graph_matching', allow_undeclared_parameters = True, automatically_declare_parameters_from_overrides = True)
        self.gm = GraphMatcher(self.get_logger())
        self.sgr_node = SituationalGraphReasoningNode()
        self.online_planes_info = {}  # Store split plane info: {split_id: {"old_id", "center", "segment", "length", ...}}
        self.online_planes_by_original_id = {}  # Map original_id -> list of split plane infos
        self.set_interface()
        self.get_json_parameters_()
        # self.get_logger().info(f"{self.params}")
        
        self.original_planes = {}
        self.prior_original_planes = {}
        self.graphs_gnn = {}    # Stores graphs in NetworkX format for GNN
        # self.graphs_msg_cache = {}  # Store original GraphMsg for later GNN conversion

        #Use parameters from ROS2 parameter server to select with matcher to use
        self.use_pgm = self.get_parameter_or('use_pgm', Parameter('use_pgm', Parameter.Type.BOOL, False)).value
        """ use_pgm: is already set as true in the /graph_matching/config/params.yaml"""

        if self.use_pgm:
            self.setup_pgm()
            self.create_subscription(PlanesDataMsg, '/s_graphs/all_map_planes', self.all_planes_callback_wrapper, 10)
            self.get_logger().info('Using PGM matcher')
        else:
            self.get_logger().info('Classic matcher initialized. Not using PGM matcher.')

        
    def get_json_parameters_(self):
        matching_package_path = ament_index_python.get_package_share_directory("graph_matching")
        json_file_path = os.path.join(matching_package_path, "config/syntheticDS_params.json")
        with open(json_file_path) as json_file:
            self.params = json.load(json_file)
            self.get_logger().info('flag self.params invariants {}'.format(self.params["invariants"]))
            self.get_logger().info('flag self.params thresholds {}'.format(self.params["thresholds"]))


    ####################### PGM RELATED FUNCTIONS ###########################################################################################
    def setup_pgm(self):
        """To use the subprocess module you need to specify the full path to the Python executable.(ad hoc PGM_ENV)"""
        """And the full path to the PGM inference script, that is the pgm_inference_wrapper.py."""
        
        """PGM setup."""
        self.get_logger().info("PGM mode enabled (using subprocess)")


        """ Path to PGM isolated environment"""
        self.pgm_python ="/root/isolated_python_env/bin/python"


        """ Path to PGM inference script"""
        self.pgm_script = "/root/workspace/src/graph_matching_gnn/graph_matching/pgm_inference_wrapper.py"



        if not os.path.exists(self.pgm_python):
            self.get_logger().error(f"PGM Python path does not exist: {self.pgm_python}")
            raise FileNotFoundError(f"PGM Python path does not exist: {self.pgm_python}")
        if not os.path.exists(self.pgm_script):
            self.get_logger().error(f"PGM script path does not exist: {self.pgm_script}")
            raise FileNotFoundError(f"PGM script path does not exist: {self.pgm_script}")


        self.get_logger().info("PGM setup complete.")


    def load_all_pickle_graphs(self):
        """Load all previously saved pickle graph files from the graph_dicts directory."""
        pickle_dir = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
        loaded_graphs = {}

        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
            self.get_logger().info(f"Created directory: {pickle_dir}")
            return loaded_graphs

        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]

        if not pickle_files:
            self.get_logger().info("No pickle files found in the graph_dicts directory")
            return loaded_graphs

        class _CompatUnpickler(pickle.Unpickler):
            """Remap old module path 'graph_wrapper' → 'situational_graphs_wrapper'."""
            def find_class(self, module, name):
                module = module.replace('graph_wrapper', 'situational_graphs_wrapper', 1)
                return super().find_class(module, name)

        for pickle_file in pickle_files:
            file_path = os.path.join(pickle_dir, pickle_file)
            try:
                with open(file_path, 'rb') as f:
                    graph_wrapper = _CompatUnpickler(f).load()

                # Raw nx.DiGraph saved by the non-GNN code path — wrap it so
                # GraphWrapper methods are available (fixes get_total_number_nodes crash)
                if isinstance(graph_wrapper, (nx.DiGraph, nx.Graph)):
                    graph_wrapper = GraphWrapper(graph_obj=graph_wrapper)

                # nx.DiGraph.name defaults to '' — use the filename instead
                stored_name = graph_wrapper.name if hasattr(graph_wrapper, 'name') else ''
                if not stored_name:
                    graph_wrapper.set_name(pickle_file[:-4])

                graph_name = graph_wrapper.name
                self.gm.graphs[graph_name] = graph_wrapper
                loaded_graphs[graph_name] = graph_wrapper
                self.get_logger().info(f"Loaded graph: {graph_name} from {pickle_file} "
                                       f"({graph_wrapper.get_total_number_nodes()} nodes)")
            except Exception as e:
                self.get_logger().error(f"Failed to load pickle file {pickle_file}: {str(e)}")

        self.get_logger().info(f"Loaded {len(loaded_graphs)} graphs from pickle files")
        return loaded_graphs

    def _adapt_online_to_gnn_format(self, graph_wrapper):
        """Convert pre-GNN format (Plane/Finite Room nodes) to GNN format (ws/room).

        Handles three cases that break the offline pipeline:
          1. Container is a raw nx.DiGraph instead of GraphWrapper  (already fixed by
             load_all_pickle_graphs, but guarded here defensively)
          2. name is empty or wrong
          3. Nodes still have the original Plane/Finite Room types from graph_callback's
             first save path (before GNN conversion fires)

        Plane nodes are renamed '{id}_s0' and populated with center/normal/length/limits
        derived from Geometric_info.  Edges are remapped to the new IDs.  Nodes that are
        already in GNN format (type 'ws' or 'room') are kept unchanged.
        """
        inner = graph_wrapper.graph if hasattr(graph_wrapper, 'graph') else graph_wrapper
        G_out = nx.DiGraph()

        # Build old→new ID mapping (Plane nodes get a _s0 split suffix)
        id_map = {}
        for node_id in inner.nodes():
            attrs = inner.nodes[node_id]
            node_id_str = str(node_id)
            if attrs.get('type') == 'Plane' and '_s' not in node_id_str:
                id_map[node_id] = f"{node_id_str}_s0"
            else:
                id_map[node_id] = node_id_str

        for node_id, attrs in inner.nodes(data=True):
            new_id = id_map[node_id]
            t = attrs.get('type', '')
            geom = attrs.get('Geometric_info', np.zeros(6))

            if t in ('Finite Room', 'room'):
                center = [float(geom[0]), float(geom[1])]
                G_out.add_node(new_id,
                    type='room',
                    center=center,
                    normal=[0.0, 0.0],
                    length=-1.0,
                    original_type='Finite Room',
                    original_attrs={'Geometric_info': geom,
                                    'draw_pos': np.array(center),
                                    'type': 'Finite Room'},
                    Geometric_info=geom,
                    draw_pos=np.array(center))

            elif t == 'Plane':
                normal_2d = np.array([float(geom[3]), float(geom[4])]) if len(geom) >= 5 else np.array([1., 0.])
                mag = np.linalg.norm(normal_2d)
                if mag > 1e-6:
                    normal_2d = normal_2d / mag
                tangent = np.array([-normal_2d[1], normal_2d[0]])
                center_3d = np.array([float(geom[0]), float(geom[1]),
                                      float(geom[2]) if len(geom) > 2 else 0.])
                raw_len = attrs.get('length')
                if raw_len is None:
                    length = 2.0
                elif isinstance(raw_len, np.ndarray):
                    length = float(raw_len[0]) if len(raw_len) > 0 else 2.0
                else:
                    length = float(raw_len)
                half = length / 2.0
                ep1 = np.array([center_3d[0] - tangent[0]*half,
                                center_3d[1] - tangent[1]*half, 0.])
                ep2 = np.array([center_3d[0] + tangent[0]*half,
                                center_3d[1] + tangent[1]*half, 0.])
                G_out.add_node(new_id,
                    type='ws',
                    center=[float(center_3d[0]), float(center_3d[1])],
                    normal=normal_2d.tolist(),
                    length=length,
                    limits=[ep1, ep2],
                    original_type='Plane',
                    original_attrs={'Geometric_info': geom,
                                    'draw_pos': geom[:2],
                                    'type': 'Plane'},
                    original_id=str(node_id).split('_s')[0])

            else:
                # Already GNN format (ws) or unknown — keep as-is
                G_out.add_node(new_id, **attrs)

        # Remap edges to new node IDs
        for u, v in inner.edges():
            new_u = id_map.get(u, str(u))
            new_v = id_map.get(v, str(v))
            if G_out.has_node(new_u) and G_out.has_node(new_v):
                G_out.add_edge(new_u, new_v)

        adapted = GraphWrapper(graph_obj=G_out)
        adapted.set_name("Online")
        return adapted

    def match_loaded_graphs(self):
        """Load already GNN-converted graphs and run PGM matching."""
        if "Prior" not in self.gm.graphs or "Online" not in self.gm.graphs:
            self.get_logger().error("Both Prior and Online graphs must be loaded before matching")
            return

        # Loaded graphs are already in GNN DiGraph format (with split planes),
        # so just assign them directly to graphs_gnn.
        # Sanitize center/normal to plain Python lists so the PGM feature
        # builder (which uses list '+' concatenation) doesn't receive numpy
        # arrays and produce wrong-shaped tensors.
        for graph_name in ["Prior", "Online"]:
            self.graphs_gnn[graph_name] = self.gm.graphs[graph_name]
            for _, attrs in self.graphs_gnn[graph_name].graph.nodes(data=True):
                for key in ("center", "normal"):
                    if key in attrs and hasattr(attrs[key], "tolist"):
                        attrs[key] = attrs[key].tolist()
            self.get_logger().info(f"Loaded {graph_name} GNN graph: "
                                   f"{self.graphs_gnn[graph_name].graph.number_of_nodes()} nodes, "
                                   f"{self.graphs_gnn[graph_name].graph.number_of_edges()} edges")

        # Run PGM matching
        success, matches, matches_full, matches_dev = self.run_pgm_matching("Prior", "Online")

        if success and matches:
            for match in matches:
                self.get_logger().info("New consistent match!")
                for i in match:
                    self.get_logger().info(f"{i['origin_node_attrs']['type']}. "
                                           f"nodes {i['origin_node']} - {i['target_node']}. "
                                           f"score {i['score']}")

        return success, matches, matches_full, matches_dev

    # ------------------------------------------------------------------ #
    #  GROUND TRUTH & EVALUATION                                           #
    # ------------------------------------------------------------------ #

    def generate_ground_truth(self):
        """Load the manually defined ground truth from ground_truth.json.

        File location: graph_dicts/ground_truth.json
        Format:
            {
              "rooms": { "online_id": "prior_id", ... },
              "ws":    [ ["online_split_id", "prior_split_id"], ... ]
            }

        "ws" is a list of pairs (not a dict) to support many-to-many mappings
        where one online split can match multiple prior splits and vice-versa,
        and to avoid silent data loss from duplicate JSON keys.

        Returns:
            list of (online_split_id (str), prior_split_id (str)) tuples
        """
        gt_path = os.path.join(
            "/root/workspace/src/graph_matching/graph_matching/graph_dicts", "ground_truth.json")

        if not os.path.exists(gt_path):
            self.get_logger().error(f"[GT] Ground truth file not found: {gt_path}")
            return []

        with open(gt_path, "r") as f:
            raw = json.load(f)

        ground_truth = []
        skipped = []

        # Rooms: dict { online_id: prior_id } — rooms are never split
        for o_id, p_id in raw.get("rooms", {}).items():
            if p_id == "??":
                skipped.append(f"rooms/{o_id}")
                continue
            ground_truth.append((str(o_id), str(p_id)))
            self.get_logger().info(
                f"[GT] Online {str(o_id):20s} (rooms) → Prior {str(p_id)}")

        # ws: list of [online_split_id, prior_split_id] pairs
        for entry in raw.get("ws", []):
            if len(entry) != 2:
                self.get_logger().warn(f"[GT] Malformed ws entry: {entry}")
                continue
            o_id, p_id = str(entry[0]), str(entry[1])
            if p_id == "??":
                skipped.append(f"ws/{o_id}")
                continue
            ground_truth.append((o_id, p_id))
            self.get_logger().info(
                f"[GT] Online {o_id:20s} (ws)    → Prior {p_id}")

        if skipped:
            self.get_logger().warn(
                f"[GT] Skipped {len(skipped)} unannotated entries: {skipped}")

        self.get_logger().info(f"[GT] Loaded {len(ground_truth)} annotated pairs")
        return ground_truth

    def evaluate_matches(self, predicted_matches, ground_truth):
        """Compare predicted matches against ground truth and log metrics.

        Args:
            predicted_matches: flat list of match dicts produced by run_pgm_matching().
                Each dict must contain:
                  "origin_split_id": prior split node ID (str), e.g. "92077532_s4"
                  "target_split_id": online split node ID (str), e.g. "75_s0"
                  For room nodes these equal the plain room ID (e.g. "59", "129").
            ground_truth: list of (online_split_id (str), prior_split_id (str)) tuples
                as returned by generate_ground_truth().
        """
        # Build sets of (prior_split_id, online_split_id) pairs
        predicted_set = {(str(m["origin_split_id"]), str(m["target_split_id"]))
                         for m in predicted_matches}
        gt_set = {(p_id, o_id) for o_id, p_id in ground_truth}

        tp = predicted_set & gt_set
        fp = predicted_set - gt_set
        fn = gt_set - predicted_set

        precision = len(tp) / len(predicted_set) if predicted_set else 0.0
        recall    = len(tp) / len(gt_set)        if gt_set        else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        self.get_logger().info("=" * 50)
        self.get_logger().info("MATCHING EVALUATION")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  GT pairs:        {len(gt_set)}")
        self.get_logger().info(f"  Predicted pairs: {len(predicted_set)}")
        self.get_logger().info(f"  True positives:  {len(tp)}")
        self.get_logger().info(f"  False positives: {len(fp)}")
        self.get_logger().info(f"  False negatives: {len(fn)}")
        self.get_logger().info(f"  Precision: {precision:.3f}")
        self.get_logger().info(f"  Recall:    {recall:.3f}")
        self.get_logger().info(f"  F1 score:  {f1:.3f}")
        if fp:
            self.get_logger().warn(f"  Wrong predictions: {fp}")
        if fn:
            self.get_logger().warn(f"  Missed GT pairs:   {fn}")

        # Per-type breakdown: build a type lookup from predicted match dicts
        # (type comes from origin_node_attrs["type"]: "Finite Room" or "Plane")
        type_of = {(str(m["origin_split_id"]), str(m["target_split_id"])):
                   m["origin_node_attrs"].get("type", "") for m in predicted_matches}
        # Collect the set of prior IDs that the model considers rooms (used for FN classification)
        room_prior_ids = {p[0] for p in predicted_set if type_of.get(p) == "Finite Room"}
        # Also collect room prior IDs from GT pairs that are TP (type known via type_of)
        room_prior_ids |= {p[0] for p in tp if type_of.get(p) == "Finite Room"}
        room_tp = sum(1 for p in tp if type_of.get(p) == "Finite Room")
        room_fp = sum(1 for p in fp if type_of.get(p) == "Finite Room")
        # For FN (not in predicted set): use collected room_prior_ids to identify rooms
        room_fn = sum(1 for p in fn if p[0] in room_prior_ids)
        self.get_logger().info(f"  [rooms] TP={room_tp}  FP={room_fp}  FN={room_fn}")
        self.get_logger().info(f"  [walls] TP={len(tp)-room_tp}  FP={len(fp)-room_fp}  FN={len(fn)-room_fn}")
        self.get_logger().info("=" * 50)

        # # ------------------------------------------------------------------
        # # Seen-based accuracy: redundant when all GT pairs are eligible.
        # # Uncomment if you need to filter out unseen/orphan wall pairs.
        # # ------------------------------------------------------------------
        # seen_metrics = {}
        # if "Online" in self.graphs_gnn:
        #     ...

        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": list(tp), "fp": list(fp), "fn": list(fn)}

    def validate_gt_against_graphs(self, ground_truth):
        """Check whether every node ID in the ground truth actually exists in
        the loaded Prior and Online GNN graphs.

        This is the first thing to run when results are poor: if GT split IDs
        (e.g. "92077532_s4") are not present in the graph, it means the graphs
        were re-generated after the GT was annotated and the split indices
        shifted.  Those pairs will always be FN regardless of model quality.

        Args:
            ground_truth: list of (online_split_id, prior_split_id) tuples
                          as returned by generate_ground_truth().
        """
        if "Prior" not in self.graphs_gnn or "Online" not in self.graphs_gnn:
            self.get_logger().error("[GT-VAL] graphs_gnn not loaded — run match_loaded_graphs() first")
            return

        prior_nodes  = set(str(n) for n in self.graphs_gnn["Prior"].graph.nodes())
        online_nodes = set(str(n) for n in self.graphs_gnn["Online"].graph.nodes())

        self.get_logger().info("[GT-VAL] Prior  nodes: " + str(sorted(prior_nodes)))
        self.get_logger().info("[GT-VAL] Online nodes: " + str(sorted(online_nodes)))

        missing_prior, missing_online, valid_pairs = [], [], []
        for o_id, p_id in ground_truth:
            bad_p = p_id not in prior_nodes
            bad_o = o_id not in online_nodes
            if bad_p or bad_o:
                if bad_p:
                    missing_prior.append(p_id)
                if bad_o:
                    missing_online.append(o_id)
            else:
                valid_pairs.append((o_id, p_id))

        self.get_logger().info("=" * 50)
        self.get_logger().info("GROUND TRUTH VALIDATION")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  Total GT pairs:       {len(ground_truth)}")
        self.get_logger().info(f"  Pairs with valid IDs: {len(valid_pairs)}")
        self.get_logger().info(f"  Prior  IDs missing:   {len(missing_prior)}")
        self.get_logger().info(f"  Online IDs missing:   {len(missing_online)}")
        if missing_prior:
            self.get_logger().warn(f"  [GT-VAL] Prior  IDs not in graph: {sorted(set(missing_prior))}")
        if missing_online:
            self.get_logger().warn(f"  [GT-VAL] Online IDs not in graph: {sorted(set(missing_online))}")
        self.get_logger().info("=" * 50)

    def _reconstruct_and_split_prior_planes(self, graph_wrapper):
        """ split_ws() expects plane dictionaries with explicit center, segment endpoints, length, and normal —
        but Prior planes arrive with just a start_point, normal, and length. so we need to reconstruct the endpoints first, then call split_ws()."""
        
        """Reconstruct segment endpoints for Prior planes from start_point/normal/length, then split them.

        Prior planes arrive with Geometric_info=[cx,cy,cz,nx,ny,nz] + length + start_point.
        We reconstruct segment endpoints from start_point and run split_ws() to match the training data format.

        Returns:
            dict: {original_plane_id_int: [split_info_dict, ...]}
        """
        planes_for_split = []
        for node_id, node_attrs in graph_wrapper.graph.nodes(data=True):
            if node_attrs.get("type") != "Plane": #Skip non-plane nodes
                continue

            # Skip planes without start_point (only Prior planes have it)
            if node_attrs.get("start_point") is None:
                continue

            geom_info = node_attrs.get("Geometric_info", np.zeros(6))
            normal_2d = geom_info[3:5] if len(geom_info) >= 5 else np.array([0., 0.])

            raw_length = node_attrs.get("length")
            if isinstance(raw_length, np.ndarray):
                length = float(raw_length[0]) if len(raw_length) > 0 else -1.0
            else:
                length = float(raw_length)

            if length <= 0:
                continue

            normal_mag = np.linalg.norm(normal_2d)
            if normal_mag < 1e-6:
                continue

            start_point = node_attrs.get("start_point")

            # Reconstruct endpoints from start_point along the tangent direction
            # Match RViz: X planes extend along +Y, Y planes extend along +X
            if abs(normal_2d[0]) > abs(normal_2d[1]):
                tangent = np.array([0.0, 1.0])
            else:
                tangent = np.array([1.0, 0.0])
            endpoint1 = np.array([start_point[0], start_point[1], 0.])
            endpoint2 = np.array([*(start_point + length * tangent), 0.])
            center_2d = (start_point + (length / 2) * tangent)

            planes_for_split.append({
                "id": int(node_id),
                "center": np.array([center_2d[0], center_2d[1], 0.]),
                "segment": [endpoint1, endpoint2],
                "length": length,
                "normal": np.array([*normal_2d, 0.]),
                "xy_type": "x" if abs(normal_2d[0]) > abs(normal_2d[1]) else "y",
                "msg": None,  # No ROS msg for Prior plane but required by split_ws()
            })

        if not planes_for_split:
            return {}

        splitted = self.sgr_node.split_ws(planes_for_split)

        splits_by_id = {}
        for plane_dict in splitted:
            orig_id = plane_dict["old_id"]
            if orig_id not in splits_by_id:
                splits_by_id[orig_id] = []
            splits_by_id[orig_id].append({
                "center": plane_dict["center"],
                "segment": plane_dict["segment"],
                "length": plane_dict["length"],
                "normal": plane_dict["normal"],
                "xy_type": plane_dict["xy_type"],
            })

        return splits_by_id

    def convert_wrapper_to_gnn_format(self, graph_wrapper):
        """Convert GraphWrapper object (that has an undirected graph) to a new GraphWrapper with DiGraph for (compatibility with PGM functions).

        Both Online and Prior planes are split into individual wall segments to match the
        training data format. 
        Online splits come from all_planes_callback_wrapper when ros messages regarding planes are received
        
        Prior splits are reconstructed from center/normal/length and split via split_ws().

        Args:
            graph_wrapper: GraphWrapper object containing an undirected nx.Graph

        Returns:
            GraphWrapper containing a DiGraph with nodes having: type, center, normal, length, original_type, original_attrs
        """
        # Convert wrapper to directed and copy into G
        graph_wrapper.to_directed()
        G = copy.deepcopy(graph_wrapper)

        # Compute correct plane centers based on graph type (before splitting)
        plane_centers = {}
        is_prior = G.name == "Prior" 
        for node_id, node_attrs in G.graph.nodes(data=True):
            if node_attrs.get("type") != "Plane":
                continue
            plane_id_int = int(node_id)
            geom_info = node_attrs.get("Geometric_info", np.zeros(6))
            normal_2d = geom_info[3:5] if len(geom_info) >= 5 else np.array([0., 0.])
            normal_mag = np.linalg.norm(normal_2d)
            if normal_mag < 1e-6:
                continue
            # Match RViz: X planes extend along +Y, Y planes extend along +X
            if abs(normal_2d[0]) > abs(normal_2d[1]):
                tangent = np.array([0.0, 1.0])
            else:
                tangent = np.array([1.0, 0.0])

            if is_prior:
                # Prior: compute center from start_point + length
                start_point = node_attrs.get("start_point")
                if start_point is None:
                    print(f"[ERROR] Prior plane {node_id} missing start_point attribute")
                    continue
                raw_length = node_attrs.get("length")
                if isinstance(raw_length, np.ndarray):
                    length = float(raw_length[0]) if len(raw_length) > 0 else -1.0
                elif raw_length is not None:
                    length = float(raw_length)
                else:
                    print(f"[ERROR] Prior plane {node_id} missing length attribute")
                    continue
                if length <= 0:
                    continue
                center_2d = start_point + (length / 2) * tangent
                endpoint1 = np.array([start_point[0], start_point[1], 0.])
                endpoint2 = np.array([*(start_point + length * tangent), 0.])
                plane_centers[plane_id_int] = {
                    "id": plane_id_int,
                    "center": np.array([center_2d[0], center_2d[1], 0.]),
                    "segment": [endpoint1, endpoint2],
                    "length": length,
                    "normal": np.array([normal_2d[0], normal_2d[1], 0.]),
                }
            else:
                # Online: use center from characterize_ws() stored in original_planes
                orig_plane = self.original_planes.get(plane_id_int)
                if orig_plane is None:
                    print(f"[ERROR] Online plane {node_id} not found in original_planes from characterize_ws()")
                    continue
                plane_centers[plane_id_int] = orig_plane

        # Store prior centers for use in splitting
        if is_prior:
            self.prior_original_planes = plane_centers

        prior_splits_by_id = self._reconstruct_and_split_prior_planes(G)

        for node_id, node_attrs in list(G.graph.nodes(data=True)):
            node_id_str = str(node_id)

            #Map types: "Finite Room" -> "room", "Plane" -> "ws" and remove unknown types
            original_type = node_attrs.get("type", "")
            if original_type == "Finite Room":
                gnn_type = "room"
            elif original_type == "Plane":
                gnn_type = "ws"
            else:
                G.graph.remove_node(node_id)  # Remove unknown types
                continue

            # Get geometry info
            geom_info = node_attrs.get("Geometric_info", np.zeros(6))

            # Store original attributes for result conversion
            original_attrs = {k: v for k, v in node_attrs.items()}

            #Construct nodes for rooms and planes (before splitting for visualization purposes, then we will split planes into segments and create one node per segment)

            #room node
            if gnn_type == "room":
                center = geom_info[:2].tolist() if len(geom_info) >= 2 else [0., 0.]
                G.graph.add_node(node_id_str,
                        type=gnn_type,
                        center=center,
                        normal=[0., 0.],
                        length=-1.0,
                        original_type=original_type,
                        original_attrs=original_attrs)

            else:#ws node
                plane_id_int = int(node_id)
                plane_info = plane_centers.get(plane_id_int)
                if plane_info is None:
                    print(f"[ERROR] Plane {node_id} has no computed center")
                    G.graph.remove_node(node_id)
                    continue
                center = plane_info["center"][:2].tolist()
                normal_vec = geom_info[3:5] if len(geom_info) >= 5 else np.array([0., 0.])
                normal_magnitude = np.linalg.norm(normal_vec)
                if normal_magnitude > 1e-6:
                        normal = (normal_vec / normal_magnitude).tolist()
                else:
                        normal = [0., 0.]
                length = float(plane_info["length"])

                # Use segment endpoints from plane_centers
                limits = np.array(plane_info["segment"])

                G.graph.add_node(node_id_str,
                            type=gnn_type,
                            center=center,
                            normal=normal,
                            length=length,
                            original_type=original_type,
                            original_attrs=original_attrs,
                            limits=limits)
        
        
        if is_prior:
            G.filterout_unparented_nodes()
                
        # G.from_2D_to_3D()
        # G._add_complete_viz_attributes_to_graph()
        # visualize_nxgraph_3d(G, G.name, visualize_alone=True, include_node_ids=True, blocking=True)    
        # G.from_3D_to_2D()

        


        ####################################################################################################################
        #Splitting planes into segments for both Online and Prior.

        # Iterate over ws nodes and split where possible
        # Save room neighbors before removing original nodes, for edge creation later
        room_neighbors_by_wall = {}
        for node_id, node_attrs in list(G.graph.nodes(data=True)):
            if node_attrs.get("type") != "ws":
                continue

            node_id_str = str(node_id)
            plane_id_int = int(node_id)

            # Check for splits — use the correct source based on graph type
            splits = None
            if is_prior:
                if plane_id_int in prior_splits_by_id:
                    splits = prior_splits_by_id[plane_id_int]
            else:
                if plane_id_int in self.online_planes_by_original_id:
                    splits = self.online_planes_by_original_id[plane_id_int]

            if splits:
                # Save room neighbors for that wall before removing this node
                neighbourhood = G.get_neighbourhood_graph(node_id)
                rooms_only = neighbourhood.filter_graph_by_node_types(["room"])
                room_neighbors_by_wall[node_id_str] = [                             #for that original ID you have all the rooms connected to it (neighbors in the original graph, before splitting)
                    str(rid) for rid in rooms_only.graph.nodes() if rid != node_id
                ]

                # Create one GNN node per split segment
                split_ids = []
                for si, split in enumerate(splits):
                    split_node_id = f"{node_id_str}_s{si}"
                    split_ids.append(split_node_id)

                    split_center = split["center"][:2].tolist() if isinstance(split["center"], np.ndarray) else list(split["center"][:2])

                    normal_vec = split["normal"][:2] if isinstance(split["normal"], np.ndarray) else np.array(split["normal"][:2])
                    normal_magnitude = np.linalg.norm(normal_vec)
                    if normal_magnitude > 1e-6:
                        split_normal = (normal_vec / normal_magnitude).tolist()
                    else:
                        split_normal = [0., 0.]

                    G.graph.add_node(split_node_id,
                        type="ws",
                        center=split_center,
                        normal=split_normal,
                        length=float(split["length"]),
                        original_type=node_attrs.get("original_type", "Plane"),
                        original_attrs=node_attrs.get("original_attrs", {}),
                        original_id=node_id_str,
                        limits=split["segment"])

                # Remove original plane node (replaced by split nodes)
                G.graph.remove_node(node_id)
            
        #Edges generation 
        
        # Create edges between split ws nodes and their neighboring rooms.
        
        # Group split nodes by their original wall ID
        splits_by_original = {}
        for node_id, node_attrs in G.graph.nodes(data=True):
            if node_attrs.get("type") != "ws":
                continue
            original_id = node_attrs.get("original_id")
            if original_id is None:
                continue
            splits_by_original.setdefault(original_id, []).append((node_id, node_attrs))

        # For each original wall, connect each room neighbor to the closest split node (Room -->ws)
        for original_id, split_nodes in splits_by_original.items():
            room_neighbors = room_neighbors_by_wall.get(original_id, [])

            for room_id in room_neighbors:
                room_center = np.array(G.graph.nodes[room_id]["center"])

                best_split_id = None
                best_distance = float("inf")
                for split_id, split_attrs in split_nodes:
                    split_center = np.array(split_attrs["center"])
                    dist = np.linalg.norm(room_center - split_center)
                    if dist < best_distance:
                        best_distance = dist
                        best_split_id = split_id

                if best_split_id is not None:
                    # Training data has ws_belongs_room: room → ws (unidirectional)
                    G.graph.add_edge(room_id, best_split_id)

        ####################################################################################################################
        # Room→room edges: two rooms share a wall when each owns a split ws node
        # whose center is very close to the other's — the two opposing surfaces of
        # the same physical wall.
        #
        # Algorithm:
        #   1. Collect every split ws node and map it to its room (room→ws edge).
        #   2. Log the full pairwise-distance distribution so the threshold can be tuned.
        #   3. For every pair of split ws nodes within WALL_PAIR_THRESHOLD whose normals
        #      point in roughly opposite directions, connect their parent rooms.

        # Step 1: build ws→room map — reuse splits_by_original computed above
        split_ws = [
            (nid, nattrs)
            for nodes_list in splits_by_original.values()
            for nid, nattrs in nodes_list
        ]
        ws_to_room = {}
        for nid, _ in split_ws:
            for pred in G.graph.predecessors(nid):
                if G.graph.nodes[pred].get("type") == "room":
                    ws_to_room[nid] = pred
                    break  # each split ws belongs to at most one room

        ws_center_map = {nid: np.array(attrs["center"]) for nid, attrs in split_ws}
        ws_normal_map = {nid: np.array(attrs.get("normal", [0., 0.])) for nid, attrs in split_ws}
        ws_ids_list = list(ws_center_map.keys())

        # Step 2: log pairwise distance distribution to help calibrate WALL_PAIR_THRESHOLD
        pairwise_dists = []
        for i in range(len(ws_ids_list)):
            for j in range(i + 1, len(ws_ids_list)):
                d = float(np.linalg.norm(ws_center_map[ws_ids_list[i]] - ws_center_map[ws_ids_list[j]]))
                pairwise_dists.append(d)
        if pairwise_dists:
            pairwise_sorted = sorted(pairwise_dists)
            self.get_logger().info(
                f"[WALL-PAIR] {len(pairwise_dists)} pairwise ws-ws distances | "
                f"min={pairwise_sorted[0]:.3f}  "
                f"p10={np.percentile(pairwise_dists, 10):.3f}  "
                f"p25={np.percentile(pairwise_dists, 25):.3f}  "
                f"median={np.percentile(pairwise_dists, 50):.3f}  "
                f"p75={np.percentile(pairwise_dists, 75):.3f}  "
                f"max={pairwise_sorted[-1]:.3f}")
            self.get_logger().info(
                f"[WALL-PAIR] 20 smallest: {[f'{d:.3f}' for d in pairwise_sorted[:20]]}")

        # Step 3: connect rooms whose ws faces are close enough to form a physical wall.
        # WALL_PAIR_THRESHOLD is the maximum center-to-center distance between the two
        # opposing surfaces of a wall.  Tune it using the [WALL-PAIR] log lines above:
        # there should be a clear gap between the small cluster of wall-pair distances
        # and the next group of unrelated segment distances.
        # Typical wall thickness in SLAM environments: 0.05 – 0.30 m.
        WALL_PAIR_THRESHOLD = 1.0  # meters — adjust based on [WALL-PAIR] log output

        for i in range(len(ws_ids_list)):
            for j in range(i + 1, len(ws_ids_list)):
                sid_i = ws_ids_list[i]
                sid_j = ws_ids_list[j]

                dist = float(np.linalg.norm(ws_center_map[sid_i] - ws_center_map[sid_j]))
                if dist > WALL_PAIR_THRESHOLD:
                    continue

                # The two faces of a physical wall have roughly opposite normals.
                # Skip pairs that point in the same (or perpendicular) direction.
                ni = ws_normal_map[sid_i]
                nj = ws_normal_map[sid_j]
                ni_mag = np.linalg.norm(ni)
                nj_mag = np.linalg.norm(nj)
                if ni_mag > 1e-6 and nj_mag > 1e-6:
                    dot = float(np.dot(ni / ni_mag, nj / nj_mag))
                    if dot > 0.0:  # same half-space → not opposing faces of a wall
                        continue

                room_i = ws_to_room.get(sid_i)
                room_j = ws_to_room.get(sid_j)
                if room_i is None or room_j is None or room_i == room_j:
                    continue

                if not G.graph.has_edge(room_i, room_j):
                    G.graph.add_edge(room_i, room_j)

        ####################################################################################################################
        # Add ws→ws edges per room (ws_same_room), matching the training data format.
        # Verified from original.pkl: pattern is angularly-sorted chain + shortcut.
        # For n ws nodes: n-1 chain edges + 1 shortcut = n edges total.
        #
        # # PREVIOUS APPROACH (fully connected) — does NOT match training data:
        # for node_id, node_attrs in G.graph.nodes(data=True):
        #     if node_attrs.get("type") != "room":
        #         continue
        #     ws_neighbors = [
        #         n for n in G.graph.successors(node_id)
        #         if G.graph.nodes[n].get("type") == "ws"
        #     ]
        #     if len(ws_neighbors) < 2:
        #         continue
        #     for i in range(len(ws_neighbors)):
        #         for j in range(i + 1, len(ws_neighbors)):
        #             if not G.graph.has_edge(ws_neighbors[i], ws_neighbors[j]):
        #                 G.graph.add_edge(ws_neighbors[i], ws_neighbors[j])
        for node_id, node_attrs in G.graph.nodes(data=True):
            if node_attrs.get("type") != "room":
                continue
            ws_neighbors = [
                n for n in G.graph.successors(node_id)
                if G.graph.nodes[n].get("type") == "ws"
            ]
            if len(ws_neighbors) < 2:
                continue
            # Sort ws nodes angularly around the room center so that consecutive
            # edges connect spatially adjacent walls
            room_center = np.array(node_attrs.get("center", [0., 0., 0.]))
            ws_neighbors.sort(key=lambda n: np.arctan2(
                np.array(G.graph.nodes[n]["center"])[1] - room_center[1],
                np.array(G.graph.nodes[n]["center"])[0] - room_center[0]
            ))
            # Forward chain: ws_0 → ws_1 → ws_2 → ... → ws_(n-1)
            for i in range(len(ws_neighbors) - 1):
                if not G.graph.has_edge(ws_neighbors[i], ws_neighbors[i + 1]):
                    G.graph.add_edge(ws_neighbors[i], ws_neighbors[i + 1])
            # Shortcut: ws_0 → ws_(n-1)
            if not G.graph.has_edge(ws_neighbors[0], ws_neighbors[-1]):
                G.graph.add_edge(ws_neighbors[0], ws_neighbors[-1])

        
        # # Remove isolated ws nodes (no room predecessor) for both Prior and Online.
        # # Training data has every ws connected to exactly one room — orphan ws nodes
        # # are a structural pattern the GNN was never trained on and hurt match quality.
        # orphan_ws = [
        #     n for n, d in G.graph.nodes(data=True)
        #     if d.get("type") == "ws"
        #     and not any(G.graph.nodes[p].get("type") == "room" for p in G.graph.predecessors(n))
        # ]
        # if orphan_ws:
        #     self.get_logger().warn(f"[{G.name}] Removing {len(orphan_ws)} orphan ws nodes with no room connection: {orphan_ws}")
        #     G.graph.remove_nodes_from(orphan_ws)

        # G.from_2D_to_3D()
        # G._add_complete_viz_attributes_to_graph()
        # visualize_nxgraph_3d(G, G.name, visualize_alone=True, include_node_ids=True, blocking=True)
        # G.from_3D_to_2D()

        return G



    def run_pgm_matching(self, graph1, graph2):
        """Run PGM matching on two graphs."""
        try:
            # Load graphs (access .graph to get the underlying DiGraph from wrapper)
            g1 = self.graphs_gnn[graph1].graph
            g2 = self.graphs_gnn[graph2].graph

            self.get_logger().info(f"Running PGM on {graph1} ({g1.number_of_nodes()} nodes) "
                                f"vs {graph2} ({g2.number_of_nodes()} nodes)")


            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f1:
                g1_path = f1.name
                pickle.dump(g1, f1)
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f2:
                g2_path = f2.name
                pickle.dump(g2, f2)
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f3:
                output_path = f3.name
            
            # Run subprocess (Note: susbprocess run has to return 0 for the code to proceed, otherwise it will jump to the except block and return no matches)
            result = subprocess.run(
                [self.pgm_python, self.pgm_script, g1_path, g2_path, output_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        self.get_logger().info(f"[PGM] {line}")
            
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line:
                        self.get_logger().warn(f"[PGM stderr] {line}")
            
            if result.returncode != 0:
                self.get_logger().error(f"PGM inference failed with code: {result.returncode}")
                os.unlink(g1_path)
                os.unlink(g2_path)
                os.unlink(output_path)
                return False, [], [], []
            
            # Load result
            with open(output_path, 'rb') as f:
                pgm_result = pickle.load(f)
            
            matching_matrix = pgm_result['matching_matrix']
            g1_nodes = pgm_result['g1_nodes']
            g2_nodes = pgm_result['g2_nodes']
            
            # Clean up
            os.unlink(g1_path)
            os.unlink(g2_path)
            os.unlink(output_path)
            
            # Convert matrix to classic match format for visualization
            matches = []
            rows, cols = np.where(matching_matrix > 0.7)
            
            for row, col in zip(rows, cols):
                if row >= len(g1_nodes) or col >= len(g2_nodes):
                    continue


                base_id = g1_nodes[row]
                target_id = g2_nodes[col]
                base_node = g1.nodes[base_id]
                target_node = g2.nodes[target_id]

                # Skip cross-type matches (room<->ws) — the GNN has no hard constraint preventing them
                if base_node["type"] != target_node["type"]:
                    self.get_logger().warn(f"Skipping cross-type match: {base_id} ({base_node['type']}) -> {target_id} ({target_node['type']})")
                    continue

                # Get original ROS attributes for both nodes
                base_attrs = dict(base_node.get("original_attrs", {}))
                target_attrs = dict(target_node.get("original_attrs", {}))

                # For split nodes (e.g. "42_s0"), map back to the original plane ID
                base_original_id = base_node.get("original_id", base_id)
                target_original_id = target_node.get("original_id", target_id)

                # Use the GNN node's actual segment center actual center not Geometric_info[:3] which is the closest point on the infinite plane to world origin, not the segment center. The segment center is stored in original_attrs["center"] for split nodes, and in original_attrs["Geometric_info"] for room nodes. This is important for accurate visualization and evaluation, especially for wall segments where the infinite plane center can be far from the actual segment center.
                # Geometric_info[:3] is the closest point on the infinite plane to world origin, not the segment center
                base_center = base_node.get("center", None)
                target_center = target_node.get("center", None)
                if base_center is not None and hasattr(base_center, "tolist"):
                    base_center = base_center.tolist()
                if target_center is not None and hasattr(target_center, "tolist"):
                    target_center = target_center.tolist()

                matches.append({
                    "origin_node": int(base_original_id),
                    "target_node": int(target_original_id),
                    # Split-level IDs kept for evaluation (e.g. "92077532_s4", "75_s0")
                    "origin_split_id": base_id,    # Prior split node ID
                    "target_split_id": target_id,  # Online split node ID
                    "origin_center": base_center,   # Actual segment center (not Geometric_info[:3])
                    "target_center": target_center, # Actual segment center (not Geometric_info[:3])
                    "origin_node_attrs": {
                        "type": base_node["original_type"],
                        **base_attrs
                    },
                    "target_node_attrs": {
                        "type": target_node["original_type"],
                        **target_attrs
                    },
                    "score": float(matching_matrix[row, col])
                })
            
            # # DEBUG: log raw GNN matches before dedup to diagnose missing wall matches
            # self.get_logger().info(f"[DEBUG RAW] {len(matches)} raw split-level matches from GNN:")
            # for m in matches:
            #     self.get_logger().info(
            #         f"  [RAW] {m['origin_node_attrs']['type']}: "
            #         f"split {m['origin_split_id']} (orig {m['origin_node']}) -> "
            #         f"split {m['target_split_id']} (orig {m['target_node']})"
            #     )

            # Enforce one-to-one matching at original plane level.
            # Multiple split-to-split matches can map to the same original pair, or one
            # original plane can appear in multiple pairs. Two-step fix:
            # 1. Count split-level votes per original pair → aggregate confidence
            # 2. Greedy assignment: assign in confidence order, skip already-used planes
            from collections import defaultdict
            pair_votes = defaultdict(list)
            for m in matches:
                key = (m["origin_node"], m["target_node"])
                pair_votes[key].append(m)

            candidates = sorted(pair_votes.values(), key=lambda v: len(v), reverse=True)

            used_origins = set()
            used_targets = set()
            unique_matches = []
            for vote_list in candidates:
                m = vote_list[0]
                if m["origin_node"] not in used_origins and m["target_node"] not in used_targets:
                    m["score"] = len(vote_list)  # number of split pairs agreeing on this match
                    unique_matches.append(m)
                    used_origins.add(m["origin_node"])
                    used_targets.add(m["target_node"])
            matches = unique_matches

            if matches:
                self.get_logger().info(f"PGM found {len(matches)} correspondences")
                for match in matches:
                    self.get_logger().info(
                        f"  {match['origin_node_attrs']['type']}: "
                        f"{match['origin_node']} -> {match['target_node']} "
                        f"(score: {match['score']:.3f})"
                    )
                return True, [matches], [matches], []
            else:
                self.get_logger().warn("PGM found no matches")
                return False, [], [], []
        
        except subprocess.TimeoutExpired:
            self.get_logger().error("PGM subprocess timed out")
            return False, [], [], []
        except Exception as e:
            self.get_logger().error(f"PGM matching error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, [], [], []



    ####################################################################################################################################################################
    
    
    
    def set_interface(self):
        self.graph_subscription = self.create_subscription(GraphMsg,'graph_matching/graphs', self.graph_callback, 0)
        self.unique_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/unique_match', 10)
        # self.best_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/best_match', 10)
        self.unique_match_visualization_inc_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/incremental', 10)
        self.unique_match_visualization_full_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/full', 10)
        self.unique_match_visualization_dev_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/dev', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_1_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_2_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_3_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_4_visualization', 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'graph_matching/subgraph_match', self.subgraph_match_srv_callback)
        # /s_graphs/all_map_planes is PGM-only — subscription created in __init__ when use_pgm=True


    def all_planes_callback_wrapper(self, msg):
        """Wrapper that extracts plane info, splits planes using split_ws(), and stores results."""
        """  original_planes dictionary that contains all the planes with their original info before splitting"""
        """ online_planes_info dictionary that contains all the planes with their info after splitting)"""
        """online_planes_by_original_id dictionary that maps each original plane ID to a list of its split segments info (since one original plane can be split into multiple segments)"""
        
        planes_msgs = msg.x_planes + msg.y_planes

        # Build planes_dict list for split_ws (expects list of dicts with keys: id, center, segment, length, normal, xy_type, msg)
        planes_dict = []
        for i, plane_msg in enumerate(planes_msgs):
            if len(plane_msg.plane_points) != 0:
                center, segment, length = self.sgr_node.characterize_ws(plane_msg.plane_points)
                plane_dict = {
                    "id": plane_msg.id,
                    "center": center,
                    "segment": segment,
                    "length": length,
                    "normal": np.array([plane_msg.nx, plane_msg.ny, plane_msg.nz]),
                    "xy_type": "x" if i < len(msg.x_planes) else "y",
                    "d": plane_msg.d,
                    "msg": plane_msg
                }
                planes_dict.append(plane_dict)

        self.original_planes = {plane["id"]: plane for plane in planes_dict}  # Store original planes by ID for reference
        
        # Split planes (Online map) at intersections using split_ws()
        splitted_planes = self.sgr_node.split_ws(planes_dict) #Dictionary with keys: id, old_id, center, segment, length, normal, xy_type, d, msg (ROS msg of the plane)

        # Store split planes info
        self.online_planes_info.clear()
        self.online_planes_by_original_id.clear()

        #Remove unnecessary informations of the planes
        for plane_dict in splitted_planes:
            split_info = {
                "old_id": plane_dict["old_id"],
                "center": plane_dict["center"],
                "segment": plane_dict["segment"],
                "length": plane_dict["length"],
                "normal": plane_dict["normal"],
                "xy_type": plane_dict["xy_type"]
            }
            self.online_planes_info[plane_dict["id"]] = split_info

            # Also map by original ID (one original can have multiple splits)
            orig_id = plane_dict["old_id"]
            if orig_id not in self.online_planes_by_original_id:
                self.online_planes_by_original_id[orig_id] = []
            self.online_planes_by_original_id[orig_id].append(split_info)

        self.get_logger().info(f"Online planes: {len(planes_dict)} original -> {len(splitted_planes)} after split. "
                               f"Lengths: {[f'{pid}:{info['length']:.2f}' for pid, info in list(self.online_planes_info.items())[:5]]}")


    def graph_callback(self, msg):
        
        
        self.get_logger().info('Incoming graph with name {}'.format(msg.name))

        graph = {"name" : msg.name}
        self.gm.set_parameters(self.params)
        
        #Node Construction for a given graph message 
        nodes = []
        for node_msg in msg.nodes:
            node_id = str(node_msg.id)
            node = [node_id, {}]
            attributes = {}
            for attrib_msg in node_msg.attributes:
                if attrib_msg.str_value:
                    attributes[attrib_msg.name] = attrib_msg.str_value
                elif attrib_msg.fl_value:
                    attributes[attrib_msg.name] = np.array(attrib_msg.fl_value)
                else:
                    print("Bad definition of attribute {}".format(attrib_msg.name))


                if node_msg.type == "Plane" and attrib_msg.name == "Geometric_info" and len(attributes[attrib_msg.name]) == 4:
                    attributes[attrib_msg.name] = plane_4_params_to_6_params(attributes[attrib_msg.name])

            if graph["name"] == "Prior" and node_msg.type == "Plane":
                self.get_logger().info(f"[DEBUG] Prior Plane {node_msg.id} attributes: {list(attributes.keys())}")
            if node_msg.type == "Plane":
                if "Geometric_info" not in attributes:
                    print(f"[ERROR] Plane node missing Geometric_info! Has: {list(attributes.keys())}")
                attributes["draw_pos"] = attributes.get("Geometric_info", [0, 0])[:2]
            elif node_msg.type == "Finite Room":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            elif node_msg.type == "floor":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            else:
                self.get_logger().info('Received unknown node type: {}'.format(node_msg.type))
            
            node[1] = attributes
            node[1]["type"] = node_msg.type
            nodes.append(node)
            
        graph["nodes"] = nodes

        #Edges Construction for a given graph message
        edges = []
        for edge_msg in msg.edges:
            edge = (str(edge_msg.origin_node), str(edge_msg.target_node))
            edges.append(edge)
        
        graph["edges"] = edges
        
        
        self.gm.set_graph_from_dict(graph, graph["name"]) #Takes dictionary representation of the graph and convert it into a GraphWrapper(store it with a given key(name))
        # dbg_graph = GraphWrapper(graph_def=graph)
        accapted_node_types = ["Finite Room", "Plane"]
        self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_types(accapted_node_types)
        self.gm.graphs[graph["name"]].set_name(graph["name"]) # Set the name of the graph in the wrapper for later reference (e.g. during visualization)
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}

        # Always save the latest graph to disk regardless of matcher mode.
        graph_dicts_dir = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
        os.makedirs(graph_dicts_dir, exist_ok=True)
        save_path = os.path.join(graph_dicts_dir, f"{graph['name']}.pkl")
        with open(save_path, "wb") as pickle_file:
            pickle.dump(nx.DiGraph(self.gm.graphs[graph["name"]].graph), pickle_file)
        self.get_logger().info(f"Saved {graph['name']} graph -> {save_path}")

        if self.use_pgm:
            if graph["name"] != "Prior" and not self.original_planes:
                self.get_logger().warn(f'Skipping GNN conversion for {graph["name"]}: original_planes not yet received from /s_graphs/all_map_planes')
            else:
                gnn_wrapper = self.convert_wrapper_to_gnn_format(self.gm.graphs[graph["name"]])#Wrapper in DiGraph format with split planes as nodes, ready for GNN processing and PGM matching
                self.graphs_gnn[graph["name"]] = gnn_wrapper
                self.get_logger().info(f'Converted {graph["name"]} to DiGraph format: {gnn_wrapper.graph.number_of_nodes()} nodes, {gnn_wrapper.graph.number_of_edges()} edges')

                # Always save Prior immediately.
                # Online is saved only at the matching trigger (>= 4 rooms) so the
                # pickle always reflects the state actually used for matching, not a
                # later callback where SLAM may have merged rooms back below 4.
                graph_dicts_dir = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
                os.makedirs(graph_dicts_dir, exist_ok=True)
                if graph["name"] == "Prior":
                    with open(os.path.join(graph_dicts_dir, "Prior.pkl"), "wb") as pickle_file:
                        pickle.dump(gnn_wrapper, pickle_file)
                    self.get_logger().info(f"Saved Prior graph to {graph_dicts_dir}/Prior.pkl")



        # self.gm.graphs[graph["name"]].draw(None, options, True)



        ### Filtering unparented nodes
        # self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)



        # ### Match
        room_ids = list(self.gm.graphs[graph["name"]].filter_graph_by_node_types("Finite Room").get_nodes_ids())
        self.get_logger().info(f"Number of rooms: {len(room_ids)}, IDs: {room_ids}")
        if graph["name"] == "Online" and len(room_ids)>=3:
            self.get_logger().info(f"Starting match!")
            # Save Online pickle here — every time matching fires — so the pickle
            # always holds the most recent >= 4-room state used for matching.
            if self.use_pgm and "Online" in self.graphs_gnn:
                graph_dicts_dir = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
                with open(os.path.join(graph_dicts_dir, "Online.pkl"), "wb") as pickle_file:
                    pickle.dump(self.graphs_gnn["Online"], pickle_file)
                self.get_logger().info(f"Saved Online graph to {graph_dicts_dir}/Online.pkl")

            # prior_room_nodes = list(self.gm.graphs['Prior'].filter_graph_by_node_attributes({'type': 'Finite Room'}).get_nodes_ids())
            # self.gm.graphs["Prior"].remove_nodes(["58", "57", "56", "55", "54", "53", "52"])
            ### ROOM NODES IN A-GRAPH: 51, 52, 53, 54, 55, 56, 57, 58
            if self.use_pgm:
                # # For PGM convert Prior to GNN format on-demand if not already done
                # # Prior always arrives first, so this block is redundant
                # if "Prior" not in self.graphs_gnn and "Prior" in self.gm.graphs:
                #     self.get_logger().info(f"Converting Prior graph to DiGraph format on-demand...")
                #     gnn_wrapper = self.convert_wrapper_to_gnn_format(self.gm.graphs["Prior"])
                #     self.graphs_gnn["Prior"] = gnn_wrapper
                #     self.get_logger().info(f'Converted Prior to DiGraph format: {gnn_wrapper.graph.number_of_nodes()} nodes, {gnn_wrapper.graph.number_of_edges()} edges')
                #if "Prior" not in self.graphs_gnn:
                #    self.get_logger().warn("Prior graph not yet received, skipping PGM matching")
                #    return
                success, matches, matches_full, matches_dev = self.run_pgm_matching("Prior", "Online")
            else:
                success, matches, matches_full, matches_dev = self.gm.match("Prior", "Online", add_deviations=True)
            # self.get_logger().info(f"DBG success: {success}")
            # self.get_logger().info(f"DBG matches: {matches}")
            # self.get_logger().info(f"DBG matches_full: {matches_full}")
            # self.get_logger().info(f"DBG matches_dev: {matches_dev}")
            for match in matches:
                self.get_logger().info(f"New consistent match!")
                for i in match:
                    self.get_logger().info(f"flag {i['origin_node_attrs']['type']}. nodes {i['origin_node']} - {i['target_node']}. score {i['score']}")
                self.get_logger().info(f" ")



            if success and len(matches) > 1:
                for i, match in enumerate(matches):
                    symmetry_match_msg = self.generate_match_msg(match)
                    symmetry_match_publisher = self.create_publisher(MatchMsg, f'graph_matching/symmetry_match_{i+1}', 10)
                    symmetry_match_publisher.publish(symmetry_match_msg)
                    symmetry_match_visualization_msg = self.generate_match_visualization_msg(matches[i])
                    symmetry_match_visualization_publisher = self.create_publisher(MarkerArrayMsg, f'graph_matching/symmetry_match_{i+1}_visualization', 10)
                    symmetry_match_visualization_publisher.publish(symmetry_match_visualization_msg)



            if success and len(matches) == 1:
                unique_match_msg = self.generate_match_msg(matches[0])
                # self.unique_match_publisher.publish(unique_match_msg)
                # unique_match_visualization_inc_msg = self.generate_match_visualization_msg(matches[0])
                # self.unique_match_visualization_inc_publisher.publish(unique_match_visualization_inc_msg)
                unique_match_visualization_full_msg = self.generate_match_visualization_msg(matches_full[0])
                self.unique_match_visualization_full_publisher.publish(unique_match_visualization_full_msg)
                # unique_match_visualization_dev_msg = self.generate_match_visualization_msg(matches_dev[0], match_type="deviations")
                # self.unique_match_visualization_dev_publisher.publish(unique_match_visualization_dev_msg)
                # time.sleep(999)



    def subgraph_match_srv_callback(self, request, response):
        self.get_logger().info('Graph Matching: Received match request from {} to {}'.format(request.base_graph, request.target_graph))




        def match_fn(request, response):
            if request.base_graph not in self.gm.graphs.keys() or request.target_graph not in self.gm.graphs.keys() or \
                self.gm.graphs[request.base_graph].is_empty() or self.gm.graphs[request.target_graph].is_empty():
                response.success = 3
            else:
                success, matches = self.gm.match(request.base_graph, request.target_graph)
                
                if success:
                    matches_msg = [self.generate_match_msg(match) for match in matches]
                    matches_visualization_msg = [self.generate_match_visualization_msg(match) for match in matches]
                    self.get_logger().warn('{} successful match(es) found!'.format(len(matches_msg)))
                    response.success = 0 if len(matches_msg) == 1 else 1





                else:
                    response.success = 2
                    self.get_logger().warn('Graph Matching: no good matches found!')





                if response.success == 0:
                    # self.unique_match_publisher.publish(matches_msg[0])
                    self.unique_match_visualization_inc_publisher.publish(matches_visualization_msg[0])
                # if response.success == 0 or response.success == 1:
                #     self.best_match_publisher.publish(matches_msg[0])
                #     self.best_match_visualization_publisher.publish(matches_visualization_msg[0])





            for match in matches:
                self.get_logger().info(f"flag new consistent match")
                for i in match:
                    self.get_logger().info(f"flag {i['origin_node_attrs']['type']} {i['score']}")
                self.get_logger().info(f" ")





            return response
        
        response = match_fn(request, response)
        return response
        # try:
        #     while rclpy.ok():
        #         match_fn(request, response)
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     pass
        # return






    def generate_match_msg(self, match):
        match_msg = MatchMsg()
        for edge in match:
            ### Edge
            edge_msg = EdgeMsg()
            edge_msg.origin_node = edge["origin_node"]
            edge_msg.target_node = edge["target_node"]
            attrib_msg = AttributeMsg()
            attrib_msg.name = "score"
            attrib_msg.fl_value = [edge["score"]]
            edge_msg.attributes = [attrib_msg]
            match_msg.edges.append(edge_msg)
            # graph_msg.name = str(score)





            ### Origin node
            origin_node_msg = NodeMsg()
            origin_node_msg.id = edge["origin_node"]
            origin_node_msg.type = edge["origin_node_attrs"]["type"]
            origin_node_msg.attributes = self.dict_to_attr_msg_list(edge["origin_node_attrs"])
            match_msg.basis_nodes.append(origin_node_msg)






            ### Target node
            target_node_msg = NodeMsg()
            target_node_msg.id = edge["target_node"]
            target_node_msg.type = edge["target_node_attrs"]["type"]
            target_node_msg.attributes = self.dict_to_attr_msg_list(edge["target_node_attrs"])
            match_msg.target_nodes.append(target_node_msg)





        return match_msg






    def dict_to_attr_msg_list(self, attr_dict):
        attr_list = []
        for attr_name in attr_dict.keys():
            attr_msg = AttributeMsg()
            attr_msg.name = attr_name
            if isinstance(attr_dict[attr_name], str): 
                attr_msg.str_value = attr_dict[attr_name]
            elif isinstance(attr_dict[attr_name], np.ndarray):
                attr_msg.fl_value = list(attr_dict[attr_name].astype(float))
            
            attr_list.append(attr_msg)





        return attr_list






    def test_with_prior_graph(self, graph_old):


        ### Translate old graph
        no_tra = np.array([0,0,0])
        tra = - np.array([0,4,0])
        no_rot = rotation_matrix_from_euler_degrees(0,0,0)
        rot = rotation_matrix_from_euler_degrees(0,0,90)





        graph = copy.deepcopy(graph_old)
        graph["name"] = "ONLINE"





        nodes = []
        for node in graph["nodes"]:
            attrs = node[1]
            geom_info = attrs["Geometric_info"]
            if attrs["type"] == "Plane":
                # trans_geom_info = transform_plane_definition([geom_info], no_tra, rot)[0]
                # trans_geom_info = transform_plane_definition([trans_geom_info], tra, no_rot)[0]
                trans_geom_info = transform_plane_definition([geom_info], tra, rot)[0]





            elif attrs["type"] == "Finite Room":
                trans_geom_info = transform_point([geom_info], tra, no_rot)[0]
                trans_geom_info = transform_point([trans_geom_info], no_tra, rot)[0]
            attrs["Geometric_info"] = trans_geom_info
            attrs["draw_pos"] = attrs["Geometric_info"][:2]
            node[1] = attrs
            nodes.append(node)
        graph["nodes"] = nodes
        
        self.gm.set_graph_from_dict(graph, graph["name"])
        self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)





        ### create room point centered graph
        room_node = "56"
        wall_nodes = ["7","32", "18", "12"]
        room_node = "55"
        wall_nodes = ["29","4", "30", "5"]





        for graph_basic in [graph_old, graph]:
            for node in graph_basic["nodes"]:



                if node[0] == room_node:
                    tra = - node[1]["Geometric_info"]
            rot = rotation_matrix_from_euler_degrees(0,0,0)






            local_graph = {'nodes' : [], 'edges' : []}
            local_graph["name"] = "room centered {}".format(graph_basic["name"])





            nodes = []
            for node in graph_basic["nodes"]:
                if node[0] == room_node or node[0] in wall_nodes:
                    attrs = node[1]
                    geom_info = attrs["Geometric_info"]
                    if attrs["type"] == "Plane":
                        trans_geom_info = transform_plane_definition([geom_info], tra, rot)[0]





                    elif attrs["type"] == "Finite Room":
                        trans_geom_info = transform_point([geom_info], tra, rot)[0]
                    attrs["Geometric_info"] = trans_geom_info
                    attrs["draw_pos"] = attrs["Geometric_info"][:2]
                    node[1] = attrs
                    nodes.append(node)





            local_graph["nodes"] = nodes
            self.gm.set_graph_from_dict(local_graph, local_graph["name"])
            options = {'node_color': self.gm.graphs[local_graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
            self.gm.graphs[local_graph["name"]].draw(local_graph["name"], options, True)
        
        success, matches = self.gm.match("Prior", "ONLINE")



    def generate_match_visualization_msg(self, match, match_type = "normal"):
        source_frame = "map"
        target_frame = "prior_map"
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)
        
        # can_transform = False
        # while not can_transform:
        #     can_transform = tf_buffer.can_transform(target_frame,source_frame,rclpy.time.Time(seconds=0.0)) #-rclpy.time.Duration(seconds=5.0)
        #     self.get_logger().info('can_transform {}'.format(can_transform))





        # transform = tf_buffer.lookup_transform(target_frame,source_frame,rclpy.time.Time(seconds=0.0), rclpy.time.Duration(seconds=2.0))
        # self.get_logger().info('transform{}'.format(transform))



        transform = TransformMsg()
        transform.translation.x, transform.translation.y, transform.translation.z = 10., 0., 0.
        transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w  = 0., 0., 0., 1.
        transform_stamped = TransformStampedMsg()
        transform_stamped.transform = transform
        
        marker_array = []
        for i, edge in enumerate(match):
            origin_geom = edge["origin_node_attrs"]["Geometric_info"]

            # Use actual split segment center if available (avoids using Geometric_info[:3] which
            # is the closest point on the infinite plane to world origin, not the segment center).
            # Fallback to Geometric_info[:3] for room nodes (which don't have split centers).
            if edge.get("origin_center") is not None:
                origin_point_original = list(edge["origin_center"])
                # Ensure 3D
                if len(origin_point_original) == 2:
                    origin_point_original.append(0.)
            else:
                # For Plane nodes after conversion, Geometric_info is [cx, cy, cz, nx, ny, nz]
                # Center is at indices 0:3
                # For Finite Room, Geometric_info is [cx, cy, cz, ...]
                # So we always use indices 0:3 for center
                origin_point_original = origin_geom[:3]



            point_msg = PointStampedMsg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = origin_point_original[0], origin_point_original[1], origin_point_original[2]
            point_msg.header.frame_id = source_frame
            # origin_point_translated_msg = BufferInterface().transform(point_msg, target_frame, rclpy.time.Time(seconds=0.0))





            origin_point_translated_msg = tf2_geometry_msgs.do_transform_point(point_msg, transform_stamped)
            origin_point = [origin_point_translated_msg.point.x, origin_point_translated_msg.point.y, origin_point_translated_msg.point.z]



            # Use actual split segment center if available; fallback to Geometric_info[:3]
            target_geom = edge["target_node_attrs"]["Geometric_info"]
            if edge.get("target_center") is not None:
                target_point = list(edge["target_center"])
                # Ensure 3D
                if len(target_point) == 2:
                    target_point.append(0.)
            else:
                # For Plane nodes after conversion, Geometric_info is [cx, cy, cz, nx, ny, nz]
                # Center is at indices 0:3
                # For Finite Room, Geometric_info is [cx, cy, cz, ...]
                # So we always use indices 0:3 for center
                target_point = list(target_geom[:3])
            if edge["origin_node_attrs"]["type"] == "Finite Room":
                origin_point[2] = 22.
                target_point[2] = 22.
            elif edge["origin_node_attrs"]["type"] == "Plane":
                origin_point[2] = 16.
                target_point[2] = 16.





            add_noise = False
            if add_noise:
                noise_scale = [.5,.5,1]
                noise = (np.random.rand(3) - [.5, .5, .5]) * noise_scale
                origin_point += noise
                target_point += noise
            
            marker_msg = MarkerMsg()
            header_msg = HeaderMsg()
            header_msg.frame_id = source_frame
            header_msg.stamp = self.get_clock().now().to_msg()
            marker_msg.header = header_msg
            marker_msg.ns = "match"
            marker_msg.id = i
            marker_msg.type = 4
            marker_msg.action = 0
            # marker_msg.pose = PoseMsg()
            scale_msg = Vector3Msg()
            if match_type == "deviations":
                scale = .4
            elif edge["origin_node_attrs"]["type"] == "Plane":
                scale = .05  # Thin lines for wall matches
            else:  # Finite Room
                scale = .2   # Thick lines for room matches
            scale_msg.x, scale_msg.y, scale_msg.z = scale, scale, scale
            marker_msg.scale = scale_msg
            color_msg = ColorRGBSMsg()
            if match_type == "normal":
                color_msg.r, color_msg.g, color_msg.b = np.random.uniform(0.5,1), np.random.uniform(0.5,1), np.random.uniform(0.5,1)
            elif match_type == "deviations":
                color_msg.r, color_msg.g, color_msg.b = 0., 0., 0.
            color_msg.a = 1.
            marker_msg.color = color_msg
            marker_msg.lifetime = DurationMsg()
            marker_msg.frame_locked = True
            origin_point_msg = PointMsg()
            origin_point_msg.x, origin_point_msg.y, origin_point_msg.z = origin_point[0], origin_point[1], origin_point[2]
            target_point_msg = PointMsg()
            target_point_msg.x, target_point_msg.y, target_point_msg.z = target_point[0], target_point[1], target_point[2]
            points = [origin_point_msg,target_point_msg]
            marker_msg.points = points
            marker_array.append(marker_msg)
        
        marker_array_msg = MarkerArrayMsg()
        marker_array_msg.markers = marker_array





        return marker_array_msg



def visualize_saved_graphs(save_dir):
    print(f"[visualize] Loading graphs from: {save_dir}")
    for graph_name in ("Prior", "Online"):
        pkl_path = os.path.join(save_dir, f"{graph_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[visualize] {pkl_path} not found — skipping.")
            continue
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        nx_graph = obj.graph if isinstance(obj, GraphWrapper) else obj
        g_viz = GraphWrapper(graph_obj=copy.deepcopy(nx_graph))
        g_viz.name = graph_name

        for node_id, node_attrs in list(g_viz.get_attributes_of_all_nodes()):
            gi = np.array(node_attrs.get("Geometric_info", [0.0, 0.0, 0.0]), dtype=float)
            center = gi[:3] if len(gi) >= 3 else np.array([gi[0], gi[1], 0.0])
            if node_attrs.get("type") == "Finite Room":
                node_attrs["type"] = "room"
                node_attrs["center"] = center
                g_viz.update_node_attrs(node_id, node_attrs)
            elif node_attrs.get("type") == "Plane":
                node_attrs["type"] = "ws"
                normal = gi[3:6] if len(gi) >= 6 else np.array([0.0, 0.0, 0.0])
                node_attrs["normal"] = normal
                # Use start_point + length if available (accurate segment endpoints)
                start_point = node_attrs.get("start_point")
                raw_length = node_attrs.get("length")
                if start_point is not None and raw_length is not None:
                    length = float(raw_length[0]) if isinstance(raw_length, np.ndarray) else float(raw_length)
                    tangent = np.array([-normal[1], normal[0], 0.0])
                    tn = np.linalg.norm(tangent)
                    tangent = tangent / tn if tn > 1e-6 else np.array([1.0, 0.0, 0.0])
                    sp = np.array(start_point, dtype=float)
                    ep = sp + length * tangent[:2] if len(sp) == 2 else sp + length * tangent
                    node_attrs["center"] = np.array([*(sp[:2] + (length / 2) * tangent[:2]), 0.0])
                    node_attrs["limits"] = np.array([
                        np.array([sp[0], sp[1], 0.0]),
                        np.array([ep[0], ep[1], 0.0]),
                    ])
                else:
                    # Fallback: fixed-length line perpendicular to normal
                    node_attrs["center"] = center
                    perp = np.array([-normal[1], normal[0], 0.0])
                    pn = np.linalg.norm(perp)
                    perp = perp / pn if pn > 1e-6 else np.array([1.0, 0.0, 0.0])
                    node_attrs["limits"] = np.array([center + 2.0 * perp, center - 2.0 * perp])
                g_viz.update_node_attrs(node_id, node_attrs)

        g_viz.from_2D_to_3D()
        g_viz._add_complete_viz_attributes_to_graph()
        visualize_nxgraph_3d(g_viz, graph_name, visualize_alone=True,
                             include_node_ids=True, blocking=False)
    plt.show(block=True)


def main(args=None):
    # If called with a directory path, visualize saved graphs from that directory.
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        visualize_saved_graphs(sys.argv[1])
        return

    rclpy.init(args=args)
    graph_matching_node = GraphMatchingNode()

    # Debug mode: load saved graphs and run matching without waiting for ROS messages
    debug_offline = False  # Set to True to enable offline debug mode with saved pickles
    if debug_offline:
        graph_matching_node.load_all_pickle_graphs()

        if "Online" in graph_matching_node.gm.graphs:
            online = graph_matching_node.gm.graphs["Online"]

            # Check 1: must be a GraphWrapper (has GNN methods)
            check1 = hasattr(online, 'get_total_number_nodes')
            # Check 2: name must be 'Online'
            check2 = getattr(online, 'name', '') == 'Online'
            # Check 3: all node types must be GNN format ('ws'/'room'), not pre-GNN ('Plane'/'Finite Room')
            inner = online.graph if hasattr(online, 'graph') else online
            node_types = {d.get('type') for _, d in inner.nodes(data=True)}
            check3 = bool(node_types) and not bool(node_types & {'Plane', 'Finite Room'})

            if not (check1 and check2 and check3):
                issues = []
                if not check1: issues.append('not a GraphWrapper')
                if not check2: issues.append(f"name={getattr(online, 'name', '?')!r} (expected 'Online')")
                if not check3: issues.append(f"pre-GNN node types present: {node_types}")
                graph_matching_node.get_logger().warn(
                    f"Online graph needs adaptation — {'; '.join(issues)}")
                online = graph_matching_node._adapt_online_to_gnn_format(online)
                graph_matching_node.gm.graphs["Online"] = online
                node_types_after = {d.get('type') for _, d in online.graph.nodes(data=True)}
                graph_matching_node.get_logger().info(
                    f"Online graph adapted: {online.get_total_number_nodes()} nodes, "
                    f"types: {node_types_after}")

        if "Online" not in graph_matching_node.gm.graphs:
            graph_matching_node.get_logger().warn("No Online graph found in graph_dicts, skipping debug offline mode")
            debug_offline = False

    if debug_offline:
        g = graph_matching_node.gm.graphs["Online"].graph
        rooms = [(n, d) for n, d in g.nodes(data=True) if d.get("type") == "room"]
        ws    = [(n, d) for n, d in g.nodes(data=True) if d.get("type") == "ws"]
        print(f"[DEBUG] Rooms in pickle: {len(rooms)}, WS in pickle: {len(ws)}")
        for r_id, r_attrs in rooms:
            connected_ws = list(g.successors(r_id))
            print(f"[DEBUG]   Room {r_id}: {len(connected_ws)} ws neighbors → {connected_ws}")

        # Visualize Prior and Online graphs side by side (two separate windows).
        # visualize_nxgraph_3d() always creates its own figure, so subplot layout
        # is not supported — both windows open simultaneously, blocking=False on
        # each and a final plt.show(block=True) keeps them alive together.
        for gname in ["Prior", "Online"]:
            if gname in graph_matching_node.gm.graphs:
                g_viz = copy.deepcopy(graph_matching_node.gm.graphs[gname])
                g_viz.from_2D_to_3D()
                g_viz._add_complete_viz_attributes_to_graph()
                visualize_nxgraph_3d(g_viz, gname, visualize_alone=True,
                                     include_node_ids=True, blocking=False)
        plt.show(block=True)  # block here until both windows are closed

        result = graph_matching_node.match_loaded_graphs()

        if result is not None:
            success, matches, _, _ = result
            if success and matches:
                # Flatten all match dicts across all symmetry hypotheses
                all_predicted = [m for hypothesis in matches for m in hypothesis]

                # Load manually annotated ground truth
                gt = graph_matching_node.generate_ground_truth()

                # Validate GT node IDs against actual graph nodes (run first to
                # detect stale split indices before evaluating model quality)
                graph_matching_node.validate_gt_against_graphs(gt)

                # Evaluate and print metrics
                graph_matching_node.evaluate_matches(all_predicted, gt)
            else:
                graph_matching_node.get_logger().warn("Matching returned no results — skipping evaluation.")

        graph_matching_node.destroy_node()
        rclpy.shutdown()
        return

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()