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
from graph_manager_interface.srv import SubgraphMatch as SubgraphMatchSrv

from .GraphMatcher import GraphMatcher

class GraphManagerNode(Node):

    def __init__(self):
        super().__init__('graph_manager')
        self.gm = GraphMatcher()
        self.set_interface()
        

    def set_interface(self):
        self.graph_subscription = self.create_subscription(String,'graph_topic', self.graph_callback, 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'subgraph_match_srv', self.subgraph_match_srv_callback)


    def graph_callback(self, msg):
        graph_dict = json.loads(msg.data)
        self.get_logger().info('Graph Manager: Incoming graph with name {}'.format(graph_dict["name"]))
        self.gm.setGraph(graph_dict)
        # self.gm.plotGraphByName(graph_dict["name"])


    def subgraph_match_srv_callback(self, request, response):
        self.get_logger().info('Graph Manager: Received match request from {} to {}'.format(request.base_graph, request.target_graph))
        matches, response.success = self.gm.matchCustom(request.base_graph, request.target_graph)
        if matches:
            response.match = json.dumps(matches[0])
        else:
            self.get_logger().warn('Graph Manager: match type not correct!')
        return response


def main(args=None):
    rclpy.init(args=args)
    graph_manager_node = GraphManagerNode()

    rclpy.spin(graph_manager_node)

    graph_manager_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
