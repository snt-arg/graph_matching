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

import rclpy
import time
import json
import copy
import numpy as np
import pkg_resources
from rclpy.node import Node
from .utils import *
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros.buffer_interface import BufferInterface
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

from graph_manager_msgs.srv import SubgraphMatch as SubgraphMatchSrv
from graph_manager_msgs.msg import Graph as GraphMsg
from graph_manager_msgs.msg import Match as MatchMsg
from graph_manager_msgs.msg import Node as NodeMsg
from graph_manager_msgs.msg import Edge as EdgeMsg
from graph_manager_msgs.msg import Attribute as AttributeMsg

from .GraphMatcher import GraphMatcher
from .utils import plane_4_params_to_6_params
class GraphManagerNode(Node):

    def __init__(self):
        super().__init__('graph_manager')
        self._declare_parameters()
        self.gm = GraphMatcher(self.get_logger())
        self.set_interface()
        

    def _declare_parameters(self):

        # params = self.declare_parameters(namespace='', parameters=[\
        #     Parameter('param_1', Parameter.Type.INTEGER, 123),\
        #     Parameter('levels.datatype.Plane', Parameter.Type.STRING, 'hello world')\
        #          ])
        
        self.declare_parameter('invariants.points.0.sigma', 0.)
        self.declare_parameter('invariants.points.0.epsilon', 0.)
        self.declare_parameter('invariants.points.0.mindist', 0)
        self.declare_parameter('invariants.points&normal.0.sigp', 0.)
        self.declare_parameter('invariants.points&normal.0.epsp', 0.)
        self.declare_parameter('invariants.points&normal.0.sign', 0.)
        self.declare_parameter('invariants.points&normal.0.epsn', 0.)
        self.declare_parameter('invariants.points&normal.1.sigp', 0.)
        self.declare_parameter('invariants.points&normal.1.epsp', 0.)
        self.declare_parameter('invariants.points&normal.1.sign', 0.)
        self.declare_parameter('invariants.points&normal.1.epsn', 0.)
        self.declare_parameter('thresholds.local_intralevel', 0.)
        self.declare_parameter('thresholds.local_interlevel', 0.)
        self.declare_parameter('thresholds.global', 0.)
        self.declare_parameter('dbscan.eps', 0.)
        self.declare_parameter('dbscan.min_samples', 0)
        self.declare_parameter('levels.name', ["jmhb", "dg"])
        self.declare_parameter('levels.datatype.floor', "")
        self.declare_parameter('levels.datatype.Finite Room', "")
        self.declare_parameter('levels.datatype.Plane',"df")
        self.declare_parameter('levels.clipper_invariants.floor', 0)
        self.declare_parameter('levels.clipper_invariants.Finite Room', 0)
        self.declare_parameter('levels.clipper_invariants.Plane', 0)

    def get_parameters(self):
        self.params = {"invariants" : {"points" : [{}], "points&normal" : [{}, {}]}, "thresholds" : {}, "dbscan": {}, "levels": {"datatype": {}, "clipper_invariants" : {}}}
        self.params["invariants"]["points"][0]["sigma"] = self.get_parameter('invariants.points.0.sigma').value
        self.params["invariants"]["points"][0]["epsilon"] = self.get_parameter('invariants.points.0.epsilon').value
        self.params["invariants"]["points"][0]["mindist"] = self.get_parameter('invariants.points.0.mindist').value
        self.params["invariants"]["points&normal"][0]["sigp"] = self.get_parameter('invariants.points&normal.0.sigp').value
        self.params["invariants"]["points&normal"][0]["epsp"] = self.get_parameter('invariants.points&normal.0.epsp').value
        self.params["invariants"]["points&normal"][0]["sign"] = self.get_parameter('invariants.points&normal.0.sign').value
        self.params["invariants"]["points&normal"][0]["epsn"] = self.get_parameter('invariants.points&normal.0.epsn').value
        self.params["invariants"]["points&normal"][1]["sigp"] = self.get_parameter('invariants.points&normal.1.sigp').value
        self.params["invariants"]["points&normal"][1]["epsp"] = self.get_parameter('invariants.points&normal.1.epsp').value
        self.params["invariants"]["points&normal"][1]["sign"] = self.get_parameter('invariants.points&normal.1.sign').value
        self.params["invariants"]["points&normal"][1]["epsn"] = self.get_parameter('invariants.points&normal.1.epsn').value
        self.params["thresholds"]["local_intralevel"] = self.get_parameter('thresholds.local_intralevel').value
        self.params["thresholds"]["local_interlevel"] = self.get_parameter('thresholds.local_interlevel').value
        self.params["thresholds"]["global"] = self.get_parameter('thresholds.global').value
        self.params["dbscan"]["eps"] = self.get_parameter('dbscan.eps').value
        self.params["dbscan"]["min_samples"] = self.get_parameter('dbscan.min_samples').value
        self.params["levels"]["name"] = self.get_parameter('levels.name').value
        self.params["levels"]["datatype"]["floor"] = self.get_parameter('levels.datatype.floor').value
        self.params["levels"]["datatype"]["Finite Room"] = self.get_parameter('levels.datatype.Finite Room').value
        self.params["levels"]["datatype"]["Plane"] = self.get_parameter('levels.datatype.Plane').value
        self.params["levels"]["clipper_invariants"]["floor"] = self.get_parameter('levels.clipper_invariants.floor').value
        self.params["levels"]["clipper_invariants"]["Finite Room"] = self.get_parameter('levels.clipper_invariants.Finite Room').value
        self.params["levels"]["clipper_invariants"]["Plane"] = self.get_parameter('levels.clipper_invariants.Plane').value
        
    def set_interface(self):
        self.graph_subscription = self.create_subscription(GraphMsg,'graphs', self.graph_callback, 0)
        self.unique_match_publisher = self.create_publisher(MatchMsg, 'unique_match', 10)
        self.best_match_publisher = self.create_publisher(MatchMsg, 'best_match', 10)
        self.unique_match_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'unique_match_visualization', 10)
        self.best_match_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'best_match_visualization', 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'subgraph_match', self.subgraph_match_srv_callback)


    def graph_callback(self, msg):
        self.get_logger().info('Incoming graph with name {}'.format(msg.name))
        graph = {"name" : msg.name}
        self.get_parameters()
        self.gm.set_parameters(self.params)
        nodes = []
        for node_msg in msg.nodes:
            node = [str(node_msg.id), {}]
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

            if node_msg.type == "Plane":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
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

        edges = []
        for edge_msg in msg.edges:
            edge = (str(edge_msg.origin_node), str(edge_msg.target_node))
            edges.append(edge)
        graph["edges"] = edges
        self.gm.setGraph(graph)
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.gm.graphs[graph["name"]].draw(graph["name"], options, True)
        self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.gm.graphs[graph["name"]].draw(graph["name"], options, True)

        # self.test_with_prior_graph(graph)

        if msg.name == "ONLINE" and len(self.gm.graphs[graph["name"]].graph.nodes())>0:
            success, matches = self.gm.match_custom("Prior", "ONLINE")

            if success and len(matches) > 0:
                best_match_msg = self.generate_match_msg(matches[0])
                self.best_match_publisher.publish(best_match_msg)
                best_match_visualization_msg = self.generate_match_visualization_msg(matches[0])
                self.best_match_visualization_publisher.publish(best_match_visualization_msg)
            if success and len(matches) == 1:
                unique_match_msg = self.generate_match_msg(matches[0])
                self.unique_match_publisher.publish(unique_match_msg)
                unique_match_visualization_msg = self.generate_match_visualization_msg(matches[0])
                self.unique_match_visualization_publisher.publish(unique_match_visualization_msg)


    def subgraph_match_srv_callback(self, request, response):
        self.get_logger().info('Graph Manager: Received match request from {} to {}'.format(request.base_graph, request.target_graph))
        response.success, matches, response.score = self.gm.only_walls_match_custom(request.base_graph, request.target_graph)
        
        if response.success:
            for match in matches:
                graph_msg = GraphMsg()
                for edge in match:
                    edge_msg = EdgeMsg()
                    edge_msg.origin_node = edge["origin_node"]
                    edge_msg.origin_node = edge["target_node"]
                    attrib_msg = AttributeMsg()
                    attrib_msg.name = "score"
                    attrib_msg.fl_value = [edge["score"]]
                    edge_msg.attributes = [attrib_msg]
                    graph_msg.edges.append(edge_msg)
                response.matches.append(graph_msg)
            self.get_logger().warn('At least one successful match found!')

        else:
            self.get_logger().warn('Graph Manager: no good matches found!')
        return response


    def generate_match_msg(self, match):
        match_msg = MatchMsg()
        for edge in match[1]:
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
        
        self.gm.setGraph(graph)
        self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.gm.graphs[graph["name"]].draw(graph["name"], options, True)

        ### create room point centered graph
        room_node = "56"
        wall_nodes = ["7","32", "18", "12"]
        room_node = "55"
        wall_nodes = ["29","4", "30", "5"]

        for graph_basic in [graph_old, graph]:
            self.get_logger().info('flag graph_basic {}'.format(graph_basic["name"]))
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
            self.get_logger().info('flag nodes {}'.format(nodes))

            local_graph["nodes"] = nodes
            self.gm.setGraph(local_graph)
            options = {'node_color': self.gm.graphs[local_graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
            self.gm.graphs[local_graph["name"]].draw(local_graph["name"], options, True)
        
        success, matches = self.gm.match_custom("Prior", "ONLINE")


    def generate_match_visualization_msg(self, match):
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
        for i, edge in enumerate(match[1]):
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
            scale_msg.x, scale_msg.y, scale_msg.z = .2, .2, .2
            marker_msg.scale = scale_msg
            color_msg = ColorRGBSMsg()
            color_msg.r, color_msg.g, color_msg.b, color_msg.a  = np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0], 1.
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
    graph_manager_node = GraphManagerNode()

    rclpy.spin(graph_manager_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_manager_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
