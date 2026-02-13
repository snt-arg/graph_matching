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
        self.pgm_python ="/root/PGM_env/bin/python"


        """ Path to PGM inference script"""
        self.pgm_script = "/root/workspace/src/graph_matching_gnn/graph_matching/pgm_inference_wrapper.py"



        if not os.path.exists(self.pgm_python):
            self.get_logger().error(f"PGM Python path does not exist: {self.pgm_python}")
            raise FileNotFoundError(f"PGM Python path does not exist: {self.pgm_python}")
        if not os.path.exists(self.pgm_script):
            self.get_logger().error(f"PGM script path does not exist: {self.pgm_script}")
            raise FileNotFoundError(f"PGM script path does not exist: {self.pgm_script}")


        self.get_logger().info("PGM setup complete.")



    def _reconstruct_and_split_prior_planes(self, graph_wrapper):
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

            # Skip online planes — they are already split via the callback (Online planes)
            if int(node_id) in self.online_planes_by_original_id:
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
            if start_point is None:
                print(f"[ERROR] Prior plane {node_id} missing start_point attribute")
                continue

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
            
        #G.from_2D_to_3D()
        #G._add_complete_viz_attributes_to_graph()
        #visualize_nxgraph_3d(G, G.name, visualize_alone=True, include_node_ids=False, blocking=True)    
        #plt.pause(0.5)
        #G.from_3D_to_2D()

        
        ####################################################################################################################
        #Splitting planes into segments for both Online and Prior.
        
        # Iterate over ws nodes and split where possible
        for node_id, node_attrs in list(G.graph.nodes(data=True)):
            if node_attrs.get("type") != "ws":
                continue

            node_id_str = str(node_id)
            plane_id_int = int(node_id)

            # Check for splits
            splits = None
            if plane_id_int in self.online_planes_by_original_id:
                splits = self.online_planes_by_original_id[plane_id_int]
            elif plane_id_int in prior_splits_by_id:
                splits = prior_splits_by_id[plane_id_int]

            if splits:
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
            
        G.from_2D_to_3D()
        G._add_complete_viz_attributes_to_graph()
        visualize_nxgraph_3d(G, G.name, visualize_alone=True, include_node_ids=False, blocking=True)
        #plt.pause(0.5)
        #new edges based on spatial proximity##################################

        #######################################################################
        G.from_3D_to_2D() 
        return G



    def run_pgm_matching(self, graph1, graph2):
        """Run PGM matching on two graphs."""
        try:
            # Load graphs (access .graph to get the underlying DiGraph from wrapper)
            g1 = self.graphs_gnn[graph1].graph
            g2 = self.graphs_gnn[graph2].graph




            self.get_logger().info(f"Running PGM on {graph1} ({g1.number_of_nodes()} nodes) "
                                f"vs {graph2} ({g2.number_of_nodes()} nodes)")




            # Debug: Log node information and save graphs for analysis
            self.get_logger().info(f"Graph1 nodes: {list(g1.nodes())}")
            self.get_logger().info(f"Graph2 nodes: {list(g2.nodes())}")



            # Collect statistics for comparison with training data
            centers_g1 = []
            centers_g2 = []
            normals_g1 = []
            normals_g2 = []



            for node_id in g1.nodes():
                node = g1.nodes[node_id]
                centers_g1.append(node['center'])
                if node['type'] == 'ws':
                    normals_g1.append(node['normal'])



            for node_id in g2.nodes():
                node = g2.nodes[node_id]
                centers_g2.append(node['center'])
                if node['type'] == 'ws':
                    normals_g2.append(node['normal'])



            centers_g1 = np.array(centers_g1)
            centers_g2 = np.array(centers_g2)



            self.get_logger().info(f"=== ROS DATA STATISTICS ===")
            self.get_logger().info(f"G1 centers: X range [{centers_g1[:,0].min():.2f}, {centers_g1[:,0].max():.2f}], Y range [{centers_g1[:,1].min():.2f}, {centers_g1[:,1].max():.2f}]")
            self.get_logger().info(f"G2 centers: X range [{centers_g2[:,0].min():.2f}, {centers_g2[:,0].max():.2f}], Y range [{centers_g2[:,1].min():.2f}, {centers_g2[:,1].max():.2f}]")
            if normals_g1:
                normals_g1 = np.array(normals_g1)
                self.get_logger().info(f"G1 normals magnitude: mean={np.linalg.norm(normals_g1, axis=1).mean():.4f}")
            if normals_g2:
                normals_g2 = np.array(normals_g2)
                self.get_logger().info(f"G2 normals magnitude: mean={np.linalg.norm(normals_g2, axis=1).mean():.4f}")



            # Save graphs for offline analysis (DEBUG - can be removed later)
            debug_path = '/root/workspace/debug_graphs'
            os.makedirs(debug_path, exist_ok=True)
            with open(f'{debug_path}/g1_ros.pkl', 'wb') as f:
                pickle.dump(g1, f)
            with open(f'{debug_path}/g2_ros.pkl', 'wb') as f:
                pickle.dump(g2, f)
            self.get_logger().info(f"DEBUG: Saved graphs to {debug_path}/")



            for node_id in list(g1.nodes())[:3]:  # Show first 3 nodes
                node = g1.nodes[node_id]
                self.get_logger().info(f"G1 Node {node_id}: type={node['type']}, center={node['center']}, normal={node['normal']}, length={node['length']}")
            for node_id in list(g2.nodes())[:3]:  # Show first 3 nodes
                node = g2.nodes[node_id]
                self.get_logger().info(f"G2 Node {node_id}: type={node['type']}, center={node['center']}, normal={node['normal']}, length={node['length']}")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f1:
                g1_path = f1.name
                pickle.dump(g1, f1)
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f2:
                g2_path = f2.name
                pickle.dump(g2, f2)
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f3:
                output_path = f3.name
            
            # Run subprocess
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
            rows, cols = np.where(matching_matrix > 0.5)
            
            for row, col in zip(rows, cols):
                if row >= len(g1_nodes) or col >= len(g2_nodes):
                    continue



                base_id = g1_nodes[row]
                target_id = g2_nodes[col]
                base_node = g1.nodes[base_id]
                target_node = g2.nodes[target_id]
                # Get original ROS attributes for both nodes
                base_attrs = dict(base_node.get("original_attrs", {}))
                target_attrs = dict(target_node.get("original_attrs", {}))

                # For split nodes (e.g. "42_s0"), map back to the original plane ID
                base_original_id = base_node.get("original_id", base_id)
                target_original_id = target_node.get("original_id", target_id)

                matches.append({
                    "origin_node": int(base_original_id),
                    "target_node": int(target_original_id),
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
        self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.all_planes_callback_wrapper, 10)


    def all_planes_callback_wrapper(self, msg):
        """Wrapper that extracts plane info, splits planes using split_ws(), and stores results."""
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

        # Cache the original GraphMsg for potential later use
        # self.graphs_msg_cache[msg.name] = msg

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

            print(f"[DEBUG] Node type={node_msg.type}, attributes keys: {list(attributes.keys())}")
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
        
        
        self.gm.set_graph_from_dict(graph, graph["name"]) #Takes dictionary representation of the graph and convert it to networkx.Graph (store it with a given key)
        # dbg_graph = GraphWrapper(graph_def=graph)
        accapted_node_types = ["Finite Room", "Plane"]
        self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_types(accapted_node_types)
        self.gm.graphs[graph["name"]].set_name(graph["name"]) # Set the name of the graph in the wrapper for later reference (e.g. during visualization)
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        
        
        if self.use_pgm:
            gnn_wrapper = self.convert_wrapper_to_gnn_format(self.gm.graphs[graph["name"]])
            self.graphs_gnn[graph["name"]] = gnn_wrapper
            self.get_logger().info(f'Converted {graph["name"]} to DiGraph format: {gnn_wrapper.graph.number_of_nodes()} nodes, {gnn_wrapper.graph.number_of_edges()} edges')



        # self.gm.graphs[graph["name"]].draw(None, options, True)


        # ### Save dictionary of graphs
        # self.get_logger().info(f"FLAG type(graph) {graph}")
        # json_object = json.dumps(graph)
        # with open(f"/home/adminpc/reasoning_ws/src/graph_matching/graph_dicts/{graph['name']}.json", "w") as outfile:
        #     outfile.write(json_object)



        ### Filtering unparented nodes
        # self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)



        # ### Match
        room_ids = list(self.gm.graphs[graph["name"]].filter_graph_by_node_types("Finite Room").get_nodes_ids())
        self.get_logger().info(f"Number of rooms: {len(room_ids)}, IDs: {room_ids}")
        if graph["name"] == "Online" and len(room_ids)>=4:
            self.get_logger().info(f"Starting match!")
            # prior_room_nodes = list(self.gm.graphs['Prior'].filter_graph_by_node_attributes({'type': 'Finite Room'}).get_nodes_ids())
            # self.gm.graphs["Prior"].remove_nodes(["58", "57", "56", "55", "54", "53", "52"])
            ### ROOM NODES IN A-GRAPH: 51, 52, 53, 54, 55, 56, 57, 58
            if self.use_pgm:
                # For PGM convert Prior to GNN format on-demand if not already done
                if "Prior" not in self.graphs_gnn and "Prior" in self.gm.graphs:
                    self.get_logger().info(f"Converting Prior graph to DiGraph format on-demand...")
                    gnn_wrapper = self.convert_wrapper_to_gnn_format(self.gm.graphs["Prior"])
                    self.graphs_gnn["Prior"] = gnn_wrapper
                    self.get_logger().info(f'Converted Prior to DiGraph format: {gnn_wrapper.graph.number_of_nodes()} nodes, {gnn_wrapper.graph.number_of_edges()} edges')
                if "Prior" not in self.graphs_gnn:
                    self.get_logger().warn("Prior graph not yet received, skipping PGM matching")
                    return
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
                if response.success == 0 or response.success == 1:
                    self.best_match_publisher.publish(matches_msg[0])
                    self.best_match_visualization_publisher.publish(matches_visualization_msg[0])





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



            # For Plane nodes after conversion, Geometric_info is [cx, cy, cz, nx, ny, nz]
            # Center is at indices 0:3
            # For Finite Room, Geometric_info is [cx, cy, cz, ...]
            # So we always use indices 0:3 for center
            target_geom = edge["target_node_attrs"]["Geometric_info"]
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



def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphMatchingNode()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()






if __name__ == '__main__':
    main()