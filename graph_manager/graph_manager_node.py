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
import json
from rclpy.node import Node

from std_msgs.msg import String
from ros2_graph_manager_interface.srv import SubgraphMatch as SubgraphMatchSrv
from ros2_graph_manager_interface.msg import Graph as GraphMsg
# from ros2_graph_manager_interface.msg import Node as NodeMsg
from ros2_graph_manager_interface.msg import Edge as EdgeMsg
from ros2_graph_manager_interface.msg import Attribute as AttributeMsg

from .GraphMatcher import GraphMatcher

class GraphManagerNode(Node):

    def __init__(self):
        super().__init__('graph_manager')
        self.gm = GraphMatcher(self.get_logger())
        self.set_interface()
        

    def set_interface(self):
        self.graph_subscription = self.create_subscription(GraphMsg,'graphs', self.graph_callback, 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'subgraph_match_srv', self.subgraph_match_srv_callback)


    def graph_callback(self, msg):
        self.get_logger().info('Graph Manager: Incoming graph with name {}'.format(msg.name))
        graph = {"name" : msg.name}

        nodes = []
        for node_msg in msg.nodes:
            node = [str(node_msg.id), {}]
            attributes = {}
            for attrib_msg in node_msg.attributes:
                if attrib_msg.str_value:
                    attributes[attrib_msg.name] = attrib_msg.str_value
                elif attrib_msg.fl_value:
                    attributes[attrib_msg.name] = attrib_msg.fl_value
                else:
                    print("Bad definition of attribute {}".format(attrib_msg.name))
            
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
        # self.gm.graphs[graph["name"]].draw(graph["name"], None, True)


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


def main(args=None):
    rclpy.init(args=args)
    graph_manager_node = GraphManagerNode()

    rclpy.spin(graph_manager_node)

    graph_manager_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
