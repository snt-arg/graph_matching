import numpy as np
from .GraphMatcher import GraphMatcher
import rclpy
import copy
import time
from rclpy.node import Node
from std_msgs.msg import String


testing_mode = "node" # node / library

if testing_mode == "library":
    from .. import srv as SubgraphMatchSrv
elif testing_mode == "node":
    from graph_manager_msgs.srv import SubgraphMatch as SubgraphMatchSrv
    from graph_manager_msgs.msg import Graph as GraphMsg
    from graph_manager_msgs.msg import Node as NodeMsg
    from graph_manager_msgs.msg import Edge as EdgeMsg
    from graph_manager_msgs.msg import Attribute as AttributeMsg


ROOM_POINT_NOISE_LIMITS = [-0.01,0.01]
WALL_POINT_NOISE_LIMITS = [-0.01,0.01]
WALL_NORMAL_NOISE_LIMITS = [-0.02,0.02]

ROOM_POINT_NOISE_LIMITS = [0,0]
WALL_POINT_NOISE_LIMITS = [0,0]
WALL_NORMAL_NOISE_LIMITS = [0,0]

REAL_TRANSLATION_NOISE_LIMITS = [[-10,10],[-0.02,0.02]]
REAL_TRASLATION = np.around(np.random.uniform(REAL_TRANSLATION_NOISE_LIMITS[0][0],REAL_TRANSLATION_NOISE_LIMITS[0][1],2), decimals = 2)
REAL_ROTATION = np.around(np.random.uniform(REAL_TRANSLATION_NOISE_LIMITS[1][0],REAL_TRANSLATION_NOISE_LIMITS[1][1],2), decimals = 2)
REAL_TRANSF = [REAL_TRASLATION, REAL_ROTATION]



### Geometric functions to create the data
def translate(original, transformation): #TODO: add rotation
    new = np.array(copy.deepcopy(original))
    if len(original) == 3:
        new += np.array(transformation)
    
    elif len(original) == 6:
        new[:3] += np.array(transformation)
    return new

def generateRandomPlaneByNormalAndDistance():
    return(list(np.around(np.concatenate([np.random.uniform(-4,4,3),np.random.uniform(0,4,1)], axis=0), decimals = 2)))


def genRandPointNorm():
    normal = np.concatenate([np.random.uniform(0,4,2), [0]])
    normalized_normal = normal / np.sqrt((normal**2).sum())
    return(list(np.around(np.concatenate([np.random.uniform(-4,4,2), [0],normalized_normal], axis=0), decimals = 2)))


def add_noise_plane_by_point_and_normal(original):
    new = copy.deepcopy(original)
    new[:2] += REAL_TRASLATION
    position_noise = np.random.uniform(WALL_POINT_NOISE_LIMITS[0],WALL_POINT_NOISE_LIMITS[1],2)
    new[:2] += position_noise
    normal_noise = np.random.uniform(WALL_NORMAL_NOISE_LIMITS[0],WALL_NORMAL_NOISE_LIMITS[1],2)
    new[3:-1] += normal_noise
    new[3:] = new[3:] / np.sqrt((np.array(new[3:])**2).sum())
    return(np.around(new, decimals = 2))
    
def genRandPoint():
    return(list(np.around(np.concatenate([np.random.uniform(-1,1,2), [0]]), decimals = 2)))


def add_noise_point(original):
    new = copy.deepcopy(original)
    new[:2] += REAL_TRASLATION
    position_noise = np.random.uniform(ROOM_POINT_NOISE_LIMITS[0],ROOM_POINT_NOISE_LIMITS[1],2)
    new[:2] += position_noise
    return(np.around(new, decimals = 2))

### Definition of S_Graph from BIM information

bim_nodes_floors_attrs = [(1, {"type": "floor", "Geometric_info": [0.,0.,0.]})]

#### RANDOM
# bim_nodes_rooms_attrs = [("room_1", {"type": "Finite Room", "Geometric_info": genRandPoint()}), ("room_2", {"type": "Finite Room", "Geometric_info": genRandPoint()}), ("room_3", {"type": "Finite Room", "Geometric_info": genRandPoint()})]
# bim_nodes_walls_attrs = [("wall_1", {"type": "Plane", "Geometric_info": translate(genRandPointNorm(), bim_nodes_rooms_attrs[0][1]["Geometric_info"][0:3])}), ("wall_2", {"type": "Plane", "Geometric_info": translate(genRandPointNorm(), bim_nodes_rooms_attrs[0][1]["Geometric_info"][0:3])}),("wall_3", {"type": "Plane", "Geometric_info": translate(genRandPointNorm(), bim_nodes_rooms_attrs[0][1]["Geometric_info"][0:3])}),\
#                         ("wall_4", {"type": "Plane", "Geometric_info": translate(genRandPointNorm(), bim_nodes_rooms_attrs[0][1]["Geometric_info"][0:3])}),("wall_5", {"type": "Plane", "Geometric_info": genRandPointNorm()}), ("wall_6", {"type": "Plane", "Geometric_info": genRandPointNorm()}),\
#                         ("wall_7", {"type": "Plane", "Geometric_info": genRandPointNorm()}), ("wall_8", {"type": "Plane", "Geometric_info": genRandPointNorm()}),("wall_9", {"type": "Plane", "Geometric_info": genRandPointNorm()}),\
#                         ("wall_10",{"type": "Plane", "Geometric_info": genRandPointNorm()}),("wall_11", {"type": "Plane", "Geometric_info": genRandPointNorm()}), ("wall_12", {"type": "Plane", "Geometric_info": genRandPointNorm()})]

#### MANUAL
bim_nodes_rooms_attrs = [(101, {"type": "Finite Room", "Geometric_info": [0.1,0.,0.]}),(102, {"type": "Finite Room", "Geometric_info": [5.,5.,0.]}),(103, {"type": "Finite Room", "Geometric_info": [10.,0.,0.]})]

bim_nodes_walls_attrs = [(201, {"type": "Plane", "Geometric_info": [0.,-1.,0.,2]}), (202, {"type": "Plane", "Geometric_info": [0.,1.,0.,2.]}), (203, {"type": "Plane", "Geometric_info": [1.,0.,0.,2.]}),(204, {"type": "Plane", "Geometric_info": [-1.,0.,0.,2.]}),\
                         (205, {"type": "Plane", "Geometric_info": [0.,-1.,0.,7.]}), (206, {"type": "Plane", "Geometric_info": [0.,1.,0.,-3.]}), (207, {"type": "Plane", "Geometric_info": [1.,0.,0.,-3.]}), (208, {"type": "Plane", "Geometric_info": [-1.,0.,0.,7.]}),\
                         (209, {"type": "Plane", "Geometric_info": [0.,-1.,0.,2.]}), (210,{"type": "Plane", "Geometric_info": [0.,1.,0.,2.]}),(211,{"type": "Plane", "Geometric_info": [1.,0.,0.,-8.]}),(212, {"type": "Plane", "Geometric_info": [-1.,0.,0.,12]})]
####

bim_nodes_attrs = bim_nodes_floors_attrs
bim_nodes_attrs += bim_nodes_rooms_attrs
bim_nodes_attrs += bim_nodes_walls_attrs

bim_edges_floors_attrs = [(1,101),(1,102),(1, 103)]
bim_edges_rooms_attrs = [(101,201),(101,202),(101,203),(101,204),(102,205),(102,206),(102,207),(102,208),(103,209),(103,210),(103,211),(103,212)]
# bim_edges_interwalls_attrs = [("wall_1","wall_2"),("wall_2","wall_3")]

bim_edges_attrs = bim_edges_floors_attrs
bim_edges_attrs += bim_edges_rooms_attrs
# bim_edges_attrs += bim_edges_interwalls_attrs

bim_graph = {'name' : 'Prior', 'nodes' : bim_nodes_attrs, 'edges' : bim_edges_attrs}

# gm.setGraph(bim_graph)

bim_plot_options = {
    'node_color': 'blue',
    'node_size': 50,
    'width': 2,
    'with_labels' : True,
}
# gm.graphs["bim"].draw("bim", options = bim_plot_options, show = True)


### Definition of S_Graph from real robot information

#### Option 1 room copied
real_nodes_floors_attrs = [(1, bim_nodes_floors_attrs[0][1])]
real_nodes_rooms_attrs = [(101, bim_nodes_rooms_attrs[0][1])]
real_nodes_walls_attrs = [(201, bim_nodes_walls_attrs[0][1]), (202, bim_nodes_walls_attrs[1][1]),(203, bim_nodes_walls_attrs[2][1]),(204, bim_nodes_walls_attrs[3][1])]

real_edges_floors_attrs = [(1,101)]
real_edges_rooms_attrs = [(101,201),(101,202),(101,203),(101,204)]

# #### Option 2 rooms copied
# real_nodes_floors_attrs = [("floor_1", bim_nodes_floors_attrs[0][1])]
# real_nodes_rooms_attrs = [("room_1", bim_nodes_rooms_attrs[0][1]), ("room_2", bim_nodes_rooms_attrs[1][1]), ("room_3", bim_nodes_rooms_attrs[2][1])]
# real_nodes_walls_attrs = [("wall_1", bim_nodes_walls_attrs[0][1]), ("wall_2", bim_nodes_walls_attrs[1][1]),("wall_3", bim_nodes_walls_attrs[2][1]),\
#                           ("wall_4", bim_nodes_walls_attrs[4][1]), ("wall_5", bim_nodes_walls_attrs[5][1]), ("wall_6", bim_nodes_walls_attrs[6][1])]

# for node in real_nodes_rooms_attrs:
#     node[1]["Geometric_info"] = add_noise_point(node[1]["Geometric_info"])

# for node in real_nodes_walls_attrs:
#     node[1]["Geometric_info"] = add_noise_plane_by_point_and_normal(node[1]["Geometric_info"])



# real_edges_floors_attrs = [("floor_1", "room_1"),("floor_1", "room_2")]#,("floor_1", "room_3")]
# real_edges_rooms_attrs = [("room_1","wall_1"),("room_1","wall_2"),("room_1","wall_3"),("room_2","wall_4"),("room_2","wall_5"),("room_2","wall_6")]
# # real_edges_interwalls_attrs = [("wall_1","wall_2"),("wall_2","wall_3")]


real_nodes_attrs = real_nodes_floors_attrs
real_nodes_attrs += real_nodes_rooms_attrs
real_nodes_attrs += real_nodes_walls_attrs
real_edges_attrs = real_edges_floors_attrs
real_edges_attrs += real_edges_rooms_attrs
# real_edges_attrs += real_edges_interwalls_attrs

real_graph = {'name' : 'ONLINE','nodes' : real_nodes_attrs, 'edges' : real_edges_attrs}

# gm.setGraph(real_graph)

real_plot_options = {
    'node_color': 'red',
    'node_size': 50,
    'width': 2,
    'with_labels' : True,
}
# gm.graphs["real"].draw("bim", options = real_plot_options, show = True)


### Subgraph isomorphism matching
# gm.matchByNodeType("bim", "real", draw = True)


### Full process comparing BIM and REAL graphs

# gm.matchCustom("bim", "real")
# gm.new_match_custom("bim", "real")

if testing_mode == "library":
    gm = GraphMatcher()
    gm.setGraph(bim_graph)
    gm.graphs["bim"].draw("bim", options = bim_plot_options, show = True)
    gm.setGraph(real_graph)
    gm.graphs["real"].draw("bim", options = real_plot_options, show = True)
    gm.new_match_custom("bim", "real")

class GraphManagerTesterNode(Node):

    def __init__(self):
        super().__init__('graph_manager_tester')
        self.set_interface()
        self.send_graphs()
        time.sleep(2)
        # self.send_match_request()
        return

    
    def set_interface(self):
        self.graph_publisher = self.create_publisher(GraphMsg,'graphs', 10)
        self.match_srv_client = self.create_client(SubgraphMatchSrv, 'subgraph_match')

    
    def send_graphs(self):
        self.get_logger().info('Sending graphs')
        encoded_bim_graph = self.endecode_graph_msg(bim_graph)
        self.graph_publisher.publish(encoded_bim_graph)

        encoded_real_graph = self.endecode_graph_msg(real_graph)
        self.graph_publisher.publish(encoded_real_graph)


    def send_match_request(self):
        request = SubgraphMatchSrv.Request()
        request.base_graph = "Prior"
        request.target_graph = "Online"
        future = self.match_srv_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().warn('Response from Graph Matcher node received!')
        # self.get_logger().info('Received Match service response: {}'.format(future.result()))
        return future.result()


    def endecode_graph_msg(self, graph_dict):
        graph_msg = GraphMsg()
        graph_msg.name = graph_dict["name"]

        nodes = []
        for node in graph_dict["nodes"]:
            node_msg = NodeMsg()
            node_msg.id = node[0]
            node_msg.type = node[1]["type"]
            
            attrib_msgs = []
            for key in node[1].keys():
                attrib_msg = AttributeMsg()
                attrib_msg.name = key
                if type(node[1][key]) == str:
                    attrib_msg.str_value = node[1][key]
                    attrib_msgs.append(attrib_msg)
                elif type(node[1][key]) == list:
                    attrib_msg.fl_value =  list(map(float, node[1][key]))
                    attrib_msgs.append(attrib_msg)
                else:
                    print("Bad definition of attribute {}".format(key))
                
            node_msg.attributes = attrib_msgs
            nodes.append(node_msg)
        graph_msg.nodes = nodes

        edges = []
        for edge in graph_dict["edges"]:
            edge_msg = EdgeMsg()
            edge_msg.origin_node = edge[0]
            edge_msg.target_node = edge[1]
            edges.append(edge_msg)
        graph_msg.edges = edges
    
        return(graph_msg)

def main(args=None):
    rclpy.init(args=args)
    graph_manager_node = GraphManagerTesterNode()

    rclpy.spin(graph_manager_node)

    graph_manager_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    if testing_mode == "node":
        main()