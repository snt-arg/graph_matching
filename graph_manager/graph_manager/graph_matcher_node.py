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
from .msg import Sgraph as SgraphMsg
from .srv import SubgraphMatch as SubgraphMatchSrv

from GraphManager import GraphManager


class GraphManagerNode(Node):

    def __init__(self):
        super().__init__('graph_manager')
        self.graph_manager = GraphManager()
        self.set_interface()
        

    def set_interface(self):
        self.bim_sgraph_subscription = self.create_subscription(SgraphMsg,'bim_sgraph_topic', self.bim_sgraph_callback, 10)
        self.real_sgraph_subscription = self.create_subscription(SgraphMsg,'real_sgraph_topic', self.real_sgraph_callback, 10)

        ### self.match_publisher_ = self.create_publisher(MatchMsg, 'match_topic', 10)

        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'subgraph_match_srv', self.subgraph_match_callback)


    def bim_sgraph_callback(self):
        pass


    def real_sgraph_callback(self):
        pass


    def subgraph_match_callback(self, request, response):
        return response


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
