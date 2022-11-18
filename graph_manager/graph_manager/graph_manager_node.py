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
from rclpy.node import Node

from std_msgs.msg import String
from graph_manager_interface.msg import Sgraph as SgraphMsg
from graph_manager_interface.msg import Edge as EdgeMsg
from graph_manager_interface.srv import SubgraphMatch as SubgraphMatchSrv

from .GraphManager import GraphManager

class GraphManagerNode(Node):

    def __init__(self):
        super().__init__('graph_manager')
        self.gm = GraphManager()
        self.set_interface()
        

    def set_interface(self):
        self.bim_sgraph_subscription = self.create_subscription(SgraphMsg,'bim_sgraph_topic', self.bim_sgraph_callback, 10)
        self.real_sgraph_subscription = self.create_subscription(SgraphMsg,'real_sgraph_topic', self.real_sgraph_callback, 10)

        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'subgraph_match_srv', self.subgraph_match_callback)


    def bim_sgraph_callback(self, msg):
        nodes_attrs, edges_attrs = self.decode_graph_msg(msg)
        self.gm.setGraph("bim", nodes_attrs, edges_attrs)


    def real_sgraph_callback(self, msg):
        nodes_attrs, edges_attrs = self.decode_graph_msg(msg)
        self.gm.setGraph("real", nodes_attrs, edges_attrs)


    def decode_graph_msg(self, msg):
        nodes_attrs = [(node.id, {"type" : node.type, "pos" : node.attribues}) for node in msg.node_list()]
        edges_attrs = [(edge.origin_node, edge.target_node) for edge in msg.edge_list()]
        return nodes_attrs, edges_attrs


    def subgraph_match_callback(self, request, response):
        if request.match_type == 1: ### TODO implement this?
            match = self.gm.matchIsomorphism(request.base_graph, request.target_graph)
        elif request.match_type == 2:
            matches = self.gm.matchByNodeType(request.base_graph, request.target_graph, draw = False)
            if matches:
                response.success = True
                response.match = self.encode_edge_list(matches[0])
            else:
                response.success = False
        elif request.match_type == 3:
            response.success, matches = self.gm.matchCustom(request.base_graph, request.target_graph)
            response.match = self.encode_edge_list(matches[0])
        else:
            self.get_logger().warn('Match type not correct')
        return response

    
    def encode_edge_list(self, edges):
        # edges_msg = []
        # for edge in edges:
        #     edge_msg = EdgeMsg()
        #     edge_msg.origin_node = edge[0]
        #     edge_msg.target_node = edge[1]
        #     edges_msg.append()
        # return edges_msg

        return [EdgeMsg(edge[0], edge[1]) for edge in edges]


def main(args=None):
    rclpy.init(args=args)
    graph_manager_node = GraphManagerNode()

    rclpy.spin(graph_manager_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    graph_manager_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
