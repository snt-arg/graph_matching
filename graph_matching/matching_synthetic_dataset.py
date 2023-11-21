import matplotlib.pyplot as plt
import sys, json, os, copy
import numpy as np
import time

from GraphMatcher import GraphMatcher

reasoning_msgs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(reasoning_msgs,"config", "syntheticDS_params_synthetic.json")) as f:
    syntheticDS_params = json.load(f)

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph

with open(os.path.join(os.path.dirname(synthetic_datset_dir),"config", "graph_matching.json")) as f:
    synteticdataset_settings = json.load(f)


### GENERATE DATASET

mode = "matching"
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings)
dataset_generator.create_dataset()
# # room_clustering_dataset = dataset_generator.get_ws2room_clustering_datalodaer()
filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])

### A-GRAPH
#### ORIGINAL
if mode =="matching":
    a_graph = copy.deepcopy(filtered_nxdataset["original"][0])
#### VIEW 1
elif mode == "multiview":
    base_matrix = dataset_generator.generate_base_matrix()
    dataset_generator.graphs["views"] = [dataset_generator.generate_graph_from_base_matrix(base_matrix,add_noise= False, add_multiview=True)]
    dataset_generator.graphs["original"] = [dataset_generator.generate_graph_from_base_matrix(base_matrix,add_noise= False, add_multiview=False)]
    filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])
    a_graph = copy.deepcopy(filtered_nxdataset["views"][0].filter_graph_by_node_attributes_containted({"view" : 1}))
    visualize_nxgraph(filtered_nxdataset["original"][0], "original")
####
visualize_nxgraph(a_graph, "a_graph")
a_graph_nodes_ids = copy.deepcopy(a_graph.get_nodes_ids())
a_graph.stringify_node_ids()
a_graph.name = "A-Graph"
a_graph_plot = copy.deepcopy(a_graph)
for node_id in list(a_graph.get_nodes_ids()):
    for attr_key in copy.deepcopy(a_graph.get_attributes_of_node(node_id)):
        if attr_key not in ["type", "Geometric_info"]:
            del a_graph.graph.nodes[node_id][attr_key]

### S-GRAPH
#### ORIGINAL
if mode == "matching":

    def deviate_plane(room_id, ws_id, deviation):
        # Deviate plane
        deviated_ws_attrs = s_graph.get_attributes_of_node(ws_id)
        deviated_ws_attrs["center"] = deviated_ws_attrs["center"] + deviation
        deviated_ws_attrs["limits"][0] = deviated_ws_attrs["limits"][0] + deviation
        deviated_ws_attrs["limits"][1] = deviated_ws_attrs["limits"][1] + deviation
        deviated_ws_attrs["viz_data"] = [deviated_ws_attrs["limits"][0][:2], deviated_ws_attrs["limits"][1][:2]]
        deviated_ws_attrs["Geometric_info"] = np.concatenate([deviated_ws_attrs["center"], deviated_ws_attrs["normal"]])

        # Deviate room
        deviated_room_attrs = s_graph.get_attributes_of_node(room_id)
        deviated_room_center = np.sum([s_graph.get_attributes_of_node(ws)["center"] for ws in ws_of_deviated_room], axis = 0) / len(ws_of_deviated_room)
        deviated_room_attrs["center"] = deviated_room_center
        deviated_room_attrs["Geometric_info"] = deviated_room_center
        deviated_room_attrs["viz_data"] = deviated_room_center[:2]

    # Select nodes to deviate
    s_graph = copy.deepcopy(filtered_nxdataset["original"][0])
    deviated_room_id = list(s_graph.filter_graph_by_node_attributes_containted({"type" : "room"}).get_nodes_ids())[0]
    s_graph.get_neighbourhood_graph(deviated_room_id)
    ws_of_deviated_room = list(s_graph.get_neighbourhood_graph(deviated_room_id).filter_graph_by_node_attributes_containted({"type" : "ws"}).get_nodes_ids())
    deviated_ws_id = ws_of_deviated_room[0]
    deviation = np.array([0,0.3,0])
    deviate_plane(deviated_room_id, deviated_ws_id, deviation)


#### VIEW 2
elif mode == "multiview":
    dataset_generator.graphs["views"] = [dataset_generator.generate_graph_from_base_matrix(base_matrix,add_noise= True, add_multiview=True)]
    filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])
    s_graph = copy.deepcopy(filtered_nxdataset["views"][0].filter_graph_by_node_attributes_containted({"view" : 2}))
####

# mapping = dict(zip(s_graph.get_nodes_ids(), list(np.array(s_graph.get_nodes_ids()) + len(a_graph_nodes_ids) + 1)))
# s_graph.relabel_nodes(mapping)
visualize_nxgraph(s_graph, "s_graph")
s_graph.stringify_node_ids()
s_graph.name = "S-Graph"
s_graph_plot = copy.deepcopy(s_graph)
for node_id in list(s_graph.get_nodes_ids()):
    for attr_key in copy.deepcopy(s_graph.get_attributes_of_node(node_id)):
        if attr_key not in ["type", "Geometric_info"]:
            del s_graph.graph.nodes[node_id][attr_key]

as_graph = a_graph_plot
as_graph.add_nodes(s_graph_plot.get_attributes_of_all_nodes())
as_graph.add_edges(s_graph_plot.get_attributes_of_all_edges())

### CREATE GRAPH MATCHER

class FakeLogger(object):
    def __init__(self) -> None:
        pass
    def info(self, msg):
        print(f"FakeLogger: {msg}")
fake_logger = FakeLogger()

graph_matcher = GraphMatcher(fake_logger)
graph_matcher.set_parameters(syntheticDS_params)
graph_matcher.set_graph_from_wrapper(a_graph, "A-Graph")
graph_matcher.set_graph_from_wrapper(s_graph, "S-Graph")

# visualize_nxgraph(as_graph, "as_graph")

### MATCH
# time.sleep(99)
success, final_combinations = graph_matcher.match("A-Graph", "S-Graph")
for final_combination in final_combinations:
    print(f"flag new final combination")
    for i in final_combination:
        print(f"flag {i['origin_node_attrs']['type']} {i['score']}")

print(f"flag NUMBER OF MATCHES: {len(final_combinations)}")

if success:
    edges = []
    for edge_dict in final_combinations[0]:
        edges.append((str(edge_dict['origin_node']), str(edge_dict['target_node']), {"viz_feat" : 'pink'}))
    as_graph.add_edges(edges)
    # visualize_nxgraph(as_graph, "as_graph")
plt.show()