import matplotlib.pyplot as plt
import sys, json, os, copy

from GraphMatcher import GraphMatcher

graph_manager_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(graph_manager_dir,"config", "syntheticDS_params.json")) as f:
    syntheticDS_params = json.load(f)

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning", "graph_reasoning")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator

with open(os.path.join(synthetic_datset_dir,"..","config","SyntheticDataset", "graph_matching.json")) as f:
    synteticdataset_settings = json.load(f)


### GENERATE DATASET

dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings)
# # room_clustering_dataset = dataset_generator.get_ws2room_clustering_datalodaer()
filtered_nxdataset = dataset_generator.get_filtered_datset(["room", "ws"],["ws_belongs_room"])
a_graph = copy.deepcopy(filtered_nxdataset[0])
a_graph.stringify_node_ids()
a_graph.name = "A-Graph"
for node_id in list(a_graph.get_nodes_ids()):
    for attr_key in copy.deepcopy(a_graph.get_attributes_of_node(node_id)):
        if attr_key not in ["type", "Geometric_info"]:
            del a_graph.graph.nodes[node_id][attr_key]

s_graph = copy.deepcopy(filtered_nxdataset[0])
s_graph.stringify_node_ids()

s_graph.name = "S-Graph"
for node_id in list(s_graph.get_nodes_ids()):
    for attr_key in copy.deepcopy(s_graph.get_attributes_of_node(node_id)):
        if attr_key not in ["type", "Geometric_info"]:
            del s_graph.graph.nodes[node_id][attr_key]

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
# options = {'node_color': graph_matcher.graphs["A-Graph"].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
# graph_matcher.graphs["A-Graph"].draw("A-Graph", options, True)
graph_matcher.set_graph_from_wrapper(s_graph, "S-Graph")
# options = {'node_color': graph_matcher.graphs["S-Graph"].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
# graph_matcher.graphs["S-Graph"].draw("S-Graph", options, True)


### MATCH
graph_matcher.match("A-Graph", "S-Graph")


plt.show()