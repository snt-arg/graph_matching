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

import rclpy, os
import time
import copy
import numpy as np
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from rclpy.node import Node
from graph_matching.utils import *
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

from graph_matching.GraphMatcher import GraphMatcher
from graph_matching.utils import plane_4_params_to_6_params
class GraphMatchingNode(Node):

    def __init__(self):
        super().__init__('graph_matching', allow_undeclared_parameters = True, automatically_declare_parameters_from_overrides = True)
        self.gm = GraphMatcher(self.get_logger(), 0)    
        self.set_interface()
        self.get_json_parameters_()
        # self.get_logger().info(f"{self.params}")

        self.debug_csv_file = self.get_parameter('debug_csv_file').get_parameter_value().string_value
        if self.debug_csv_file:
            self.get_logger().info(f"Debug CSV file path: {self.debug_csv_file}")
            self.write_csv_header()
        else:
            self.get_logger().info("No debug CSV file provided.")

    def get_json_parameters_(self):
        matching_package_path = ament_index_python.get_package_share_directory("graph_matching")
        json_file_path = os.path.join(matching_package_path, "config/syntheticDS_params.json")
        with open(json_file_path) as json_file:
            self.params = json.load(json_file)
            self.get_logger().info('flag self.params invariants {}'.format(self.params["invariants"]))
            self.get_logger().info('flag self.params thresholds {}'.format(self.params["thresholds"]))

        
    def set_interface(self):
        self.graph_subscription = self.create_subscription(GraphMsg,'graph_matching/graphs', self.graph_callback, 10)
        self.unique_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/unique_match', 10)
        self.best_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/best_match', 10)
        self.unique_match_visualization_inc_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/incremental', 10)
        self.unique_match_visualization_full_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/full', 10)
        self.unique_match_visualization_dev_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/dev', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_1_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_2_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_3_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_4_visualization', 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'graph_matching/subgraph_match', self.subgraph_match_srv_callback)

    def load_all_pickle_graphs(self):
        """
        Load all previously saved pickle graph files from the graph_dicts directory.
        Returns a dictionary with graph names as keys and graph data as values.
        """
        pickle_dir = "/home/adminpc/workspace/src/graph_matching/graph_matching/graph_dicts"
        loaded_graphs = {}
        
        # Create directory if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
            self.get_logger().info(f"Created directory: {pickle_dir}")
            return loaded_graphs
        
        # Find all pickle files in the directory
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        
        if not pickle_files:
            self.get_logger().info("No pickle files found in the graph_dicts directory")
            return loaded_graphs
        
        # Load each pickle file
        for pickle_file in pickle_files:
            file_path = os.path.join(pickle_dir, pickle_file)
            try:
                with open(file_path, 'rb') as f:
                    graph_data = pickle.load(f)
                    graph_name = graph_data.get('name', pickle_file[:-4])  # Remove .pkl extension as fallback
                    loaded_graphs[graph_name] = graph_data
                    self.get_logger().info(f"Successfully loaded graph: {graph_name} from {pickle_file}")
            except Exception as e:
                self.get_logger().error(f"Failed to load pickle file {pickle_file}: {str(e)}")
        
        self.get_logger().info(f"Loaded {len(loaded_graphs)} graphs from pickle files")

        # for node in graph["nodes"]:
        #     print(f'Node ID: {node[0]}, Type: {node[1]["type"]}, Attributes: {node[1]}')
        
        plot_graph = False
        accepted_node_types = ["Finite Room", "Plane", "Door", "Window"]
        for key in loaded_graphs.keys():
            graph = loaded_graphs[key]
            self.gm.set_graph_from_dict(graph, graph["name"])
            self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_types(accepted_node_types)
            
            # if graph["name"] == "Prior":
            #     node_list = ["72","10101","90010103","90010102","10100"] #,"10110", "10120"
            #     print(self.gm.graphs[graph["name"]].get_neighbourhood_graph("10100").get_nodes_ids())
            #     print(self.gm.graphs[graph["name"]].get_neighbourhood_graph("90010103").get_nodes_ids())
            # if graph["name"] == "Online":
            #     node_list = ["174", "83", "82", "81","80"] #,"96002", "124001"
            #     print(self.gm.graphs[graph["name"]].get_neighbourhood_graph("80").get_nodes_ids())
            #     print(self.gm.graphs[graph["name"]].get_neighbourhood_graph("81").get_nodes_ids())
            # self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_list(node_list)
            
            options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}

            if plot_graph:
                self.gm.graphs[graph["name"]].draw(graph["name"], options, True)

                input("Waiting")
        return loaded_graphs

    def match_loaded_graphs(self):
        self.gm.set_parameters(self.params)
        self.gm.match("Prior", "Online", add_deviations=False)

    def graph_callback(self, msg):
        if (msg.nodes == []) or (msg.edges == []):
            self.get_logger().warn(f'Empty graph {msg.name} received, skipping...')
            return
        print("************************************************************************")
        print(f"================ Graph Matching Node: New graph {msg.name} received ================")
        print("************************************************************************")
        # self.get_logger().info('Incoming graph with name {}'.format(msg.name))
        ### !DEBUG!
        # if (msg.header.stamp.sec > 832):
        #     print("!DEBUG: Not processing graph with timestamp >832 for testing purposes")
        #     return
        graph = {"name" : msg.name}
        self.gm.set_parameters(self.params)
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
                    self.get_logger().warn("Bad definition of attribute {}".format(attrib_msg.name))

                if node_msg.type == "Plane" and attrib_msg.name == "Geometric_info" and len(attributes[attrib_msg.name]) == 4:
                    attributes[attrib_msg.name] = plane_4_params_to_6_params(attributes[attrib_msg.name])

            if node_msg.type == "Plane":
                # attributes["draw_pos"] = attributes["Geometric_info"][:2]
                attributes["draw_pos"] = self.get_plane_draw_position(node_msg, attributes, msg)
            elif node_msg.type == "Finite Room":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            elif node_msg.type == "floor":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            elif node_msg.type == "Door":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
                # print(f'Door node received: {node_id} with attributes {attributes}')
            elif node_msg.type == "Window":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
                # print(f'Window node received: {node_id} with attributes {attributes}')
            else:
                self.get_logger().warn('Received unknown node type: {}'.format(node_msg.type))
            
            node[1] = attributes
            node[1]["type"] = node_msg.type
            nodes.append(node)
            
        graph["nodes"] = nodes

        edges = []
        for edge_msg in msg.edges:
            edge = (str(edge_msg.origin_node), str(edge_msg.target_node))
            edges.append(edge)

        graph["edges"] = edges
        self.gm.set_graph_from_dict(graph, graph["name"])
        # for node in graph["nodes"]:
        #     print(f'Node ID: {node[0]}, Type: {node[1]["type"]}, Attributes: {node[1]}')
        dbg_graph = GraphWrapper(graph_def=graph)
        accepted_node_types = ["Finite Room", "Plane", "Door", "Window"]
        self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_types(accepted_node_types)
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}

        plot_graph = True
        if plot_graph:
            self.gm.graphs[graph["name"]].draw(graph["name"], options, True)
        # time.sleep(1)
        

        # ### Save dictionary of graphs
        # self.get_logger().info(f"FLAG type(graph) {graph}")
        # json_object = json.dumps(graph)
        # with open(f"/home/adminpc/reasoning_ws/src/graph_matching/graph_dicts/{graph['name']}.json", "w") as outfile:
        #     outfile.write(json_object)


        ### Filtering unparented nodes
        # self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)

        # Return if no graph Prior
        if "Prior" not in self.gm.graphs.keys() or self.gm.graphs["Prior"].is_empty():
            self.get_logger().warn('Graph Matching: no Prior graph available, skipping matching...')
            return

        # Save graph as pickle file
        # with open(f"/home/adminpc/workspace/src/graph_matching/graph_matching/graph_dicts/{graph['name']}.pkl", "wb") as pickle_file:
        #     pickle.dump(self.gm.graphs[graph["name"]], pickle_file)

        # ### Match
        room_ids = list(self.gm.graphs[graph["name"]].filter_graph_by_node_types("Finite Room").get_nodes_ids())
        self.get_logger().info(f"Number of rooms: {len(room_ids)}, IDs: {room_ids}")
        # if graph["name"] == "Online" and len(room_ids)>=2:
        if graph["name"] == "Online" and len(room_ids)>=1:
            self.get_logger().info(f"Starting match!")
            # prior_room_nodes = list(self.gm.graphs['Prior'].filter_graph_by_node_attributes({'type': 'Finite Room'}).get_nodes_ids())
            # self.gm.graphs["Prior"].remove_nodes(["58", "57", "56", "55", "54", "53", "52"])
            ### ROOM NODES IN A-GRAPH: 51, 52, 53, 54, 55, 56, 57, 58
            success, matches, matches_full, matches_dev = self.gm.match("Prior", "Online", add_deviations=True)
            # self.get_logger().info(f"DBG success: {success}")
            # self.get_logger().info(f"DBG matches: {matches}")
            # self.get_logger().info(f"DBG matches_full: {matches_full}")
            # self.get_logger().info(f"DBG matches_dev: {matches_dev}")
            if self.debug_csv_file:
                self.write_csv_line(self.gm.match_times[-1])
                print(f'{self.gm.match_times=}')

            for match in matches:
                self.get_logger().info(f"New consistent match!")
                for i in match:
                    self.get_logger().info(f"flag {i['origin_node_attrs']['type']}. nodes {i['origin_node']} - {i['target_node']}. score {i['score']}")
                self.get_logger().info(f" ")
                
            if success and len(matches) > 1:
                for i, match in enumerate(matches):
                    # symmetry_match_msg = self.generate_match_msg(match)
                    # symmetry_match_publisher = self.create_publisher(MatchMsg, f'graph_matching/symmetry_match_{i+1}', 10)
                    # symmetry_match_publisher.publish(symmetry_match_msg)
                    symmetry_match_visualization_msg = self.generate_match_visualization_msg(matches[i])
                    symmetry_match_visualization_publisher = self.create_publisher(MarkerArrayMsg, f'graph_matching/symmetry_match_{i+1}_visualization', 10)
                    symmetry_match_visualization_publisher.publish(symmetry_match_visualization_msg)

            if success and len(matches) == 1:
                print("******************* Unique match found! *******************")
                print(f"Match details:")
                print(matches[0])

                # unique_match_msg = self.generate_match_msg(matches[0])
                # self.unique_match_publisher.publish(unique_match_msg)

                # unique_match_visualization_inc_msg = self.generate_match_visualization_msg(matches[0])
                # self.unique_match_visualization_inc_publisher.publish(unique_match_visualization_inc_msg)
                unique_match_visualization_full_msg = self.generate_match_visualization_msg(matches_full[0])
                self.unique_match_visualization_full_publisher.publish(unique_match_visualization_full_msg)
                # unique_match_visualization_dev_msg = self.generate_match_visualization_msg(matches_dev[0], match_type="deviations")
                # self.unique_match_visualization_dev_publisher.publish(unique_match_visualization_dev_msg)
                # time.sleep(999)

    def write_csv_header(self):
        header = "prior_nodes,online_nodes,match_time\n"
        with open(self.debug_csv_file, 'w') as f:
            f.write(header)

    def write_csv_line(self, info_line):
        line = ''
        for info in info_line:
            line += f'{info},'
        line += '\n'

        with open(self.debug_csv_file, 'a') as f:
            f.write(line)


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
        match_msg.header.stamp = self.get_clock().now().to_msg()
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
        source_frame = "matching_map"
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
            origin_point_original = edge["origin_node_attrs"]["Geometric_info"][:3]
            point_msg = PointStampedMsg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = origin_point_original[0], origin_point_original[1], origin_point_original[2]
            point_msg.header.frame_id = source_frame
            # origin_point_translated_msg = BufferInterface().transform(point_msg, target_frame, rclpy.time.Time(seconds=0.0))

            origin_point_translated_msg = tf2_geometry_msgs.do_transform_point(point_msg, transform_stamped)
            origin_point = [origin_point_translated_msg.point.x, origin_point_translated_msg.point.y, origin_point_translated_msg.point.z]
            target_point = edge["target_node_attrs"]["Geometric_info"][:3]
            if edge["origin_node_attrs"]["type"] == "Finite Room":
                origin_point[2] = 22.
                target_point[2] = 22.
            elif edge["origin_node_attrs"]["type"] == "Plane":
                origin_point[2] = 16.
                target_point[2] = 16.

            add_noise = True
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
            if match_type == "normal":
                scale = .2
            elif match_type == "deviations":
                scale = .4
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

    def get_plane_draw_position(self, node_msg, attributes, graph_msg):
                # --- find parent room id (robust to edge direction) ---
                # node_id = str(node_msg.id)
                parent_node_id = None
                plane_draw_pos = None
                for msg_edge in graph_msg.edges:
                    # if this plane is connected, pick the other end as parent candidate
                    if str(msg_edge.target_node) == str(node_msg.id):
                        parent_node_id = str(msg_edge.origin_node)
                        break
                    if str(msg_edge.origin_node) == str(node_msg.id):
                        parent_node_id = str(msg_edge.target_node)
                        break

                if parent_node_id is None:
                    # print(f'No parent found for plane node {node_id}')
                    return attributes["Geometric_info"][:2]
                # --- get parent room center (2D) ---
                parent_draw_pos = None
                for msg_node in graph_msg.nodes:
                    if str(msg_node.id) == parent_node_id and msg_node.type == "Finite Room":
                        for attrib_msg in msg_node.attributes:
                            if attrib_msg.name == "Geometric_info" and attrib_msg.fl_value:
                                parent_draw_pos = np.array(attrib_msg.fl_value, dtype=float)[:2]
                                break
                        break

                if parent_draw_pos is None:
                    # print(f'Parent {parent_node_id} for plane node {node_id} has no Geometric_info')
                    return attributes["Geometric_info"][:2]
                else:
                    # --- plane info ---
                    gi = np.array(attributes["Geometric_info"], dtype=float)

                    p0 = gi[:2]       # closest point to origin (XY)
                    n  = gi[3:5]      # normal (XY)  (assuming layout [px,py,pz,nx,ny,nz])

                    # normalize normal (avoid divide-by-zero)
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-9:
                        # print(f'Plane node {node_id} has near-zero normal, using original draw pos')
                        adjusted = p0
                    else:
                        n_unit = n / n_norm

                        c = parent_draw_pos  # room center (XY)

                        # project room center onto plane (closest point on plane to the center)
                        # p = c - n * dot(n, (c - p0))
                        adjusted = c - n_unit * np.dot(n_unit, (c - p0))

                    return adjusted



def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphMatchingNode()
    # graph_matching_node.load_all_pickle_graphs()
    # graph_matching_node.match_loaded_graphs()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
