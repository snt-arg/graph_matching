import matplotlib.pyplot as plt
import sys, json, os, copy
import numpy as np

from GraphMatcher import GraphMatcher

graph_manager_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(graph_manager_dir,"config", "syntheticDS_params.json")) as f:
    syntheticDS_params = json.load(f)

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning", "graph_reasoning")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph

with open(os.path.join(synthetic_datset_dir,"..","config","SyntheticDataset", "graph_matching.json")) as f:
    synteticdataset_settings = json.load(f)


### GENERATE DATASET

mode = "multiview"
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings)
# # room_clustering_dataset = dataset_generator.get_ws2room_clustering_datalodaer()
filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])

### A-GRAPH
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
#### NOISE
if mode == "matching":
    s_graph = copy.deepcopy(filtered_nxdataset["noise"][0])
#### VIEW 2
elif mode == "multiview":
    dataset_generator.graphs["views"] = [dataset_generator.generate_graph_from_base_matrix(base_matrix,add_noise= True, add_multiview=True)]
    filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])
    s_graph = copy.deepcopy(filtered_nxdataset["views"][0].filter_graph_by_node_attributes_containted({"view" : 2}))
####
mapping = dict(zip(s_graph.get_nodes_ids(), list(np.array(s_graph.get_nodes_ids()) + len(a_graph_nodes_ids) + 1)))
s_graph.relabel_nodes(mapping)
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

visualize_nxgraph(as_graph, "as_graph")
### MATCH
success, final_combinations = graph_matcher.match("A-Graph", "S-Graph")
if success:
    edges = []
    for edge_dict in final_combinations[0]:
        edges.append((str(edge_dict['origin_node']), str(edge_dict['target_node']), {"viz_feat" : 'pink'}))
    as_graph.add_edges(edges)
    visualize_nxgraph(as_graph, "as_graph")
plt.show()