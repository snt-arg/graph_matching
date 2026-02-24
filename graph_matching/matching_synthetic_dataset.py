from situational_graphs_datasets.utils import get_config as get_datasets_config
from situational_graphs_datasets.graph_visualizer import visualize_nxgraph_3d
from situational_graphs_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
import matplotlib.pyplot as plt
import sys
import json
import os
import copy
import numpy as np
# import time
# from sympy import false

from GraphMatcher import GraphMatcher

syntheticDS_params_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "config", "syntheticDS_params_synthetic.json")
with open(syntheticDS_params_path) as f:
    syntheticDS_params = json.load(f)

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)

synteticdataset_settings = get_datasets_config("graph_matching")


class FakeLogger(object):
    def __init__(self) -> None:
        pass

    def info(self, msg):
        print(f"FakeLogger: {msg}")


fake_logger = FakeLogger()

# GENERATE DATASET

dataset_generator = SyntheticDatasetGenerator(
    synteticdataset_settings, logger=None, report_path="???", dataset_name="test")
dataset_generator.create_dataset()

# A-GRAPH
a_dataset = dataset_generator.extend_nxdataset(
    dataset_generator.graphs["original"], "training", "training")["train"]

visualize_nxgraph_3d(a_dataset[0], "a_graph",
                     visualize_alone=True, include_node_ids=True)

# for node_attrs in a_dataset[0].get_attributes_of_all_nodes():
#     print(node_attrs[1]["Geometric_info"])

# S-GRAPH
s_dataset = copy.deepcopy(a_dataset)

room_value = 0.0
print(f'** Room paramter set to {room_value}')
postprocess = [{"pp_name": "dropout", "room": room_value, "ws": 0.0}]
dataset_generator.settings["postprocess"]["sgraph"] = postprocess
s_dataset = dataset_generator.extend_nxdataset(
    copy.deepcopy(s_dataset), "training", "sgraph")["train"]

[s_graph.translate_geometries([5, 10, 0]) for s_graph in s_dataset]
# rotate 90 degrees around z axis
rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
[s_graph.rotate_geometries(rotation_matrix) for s_graph in s_dataset]
visualize_nxgraph_3d(s_dataset[0], "s_graph",
                     visualize_alone=True, include_node_ids=True)

# AS-GRAPH
as_dataset = []
for i in range(len(s_dataset)):
    a_graph = copy.deepcopy(a_dataset[i])
    a_graph.name = "A-Graph"
    s_graph = copy.deepcopy(s_dataset[i])
    s_graph.name = "S-Graph"

    a_graph_nodes_ids = copy.deepcopy(a_graph.get_nodes_ids())
    a_graph.stringify_node_ids()
    mapping = dict(zip(s_graph.get_nodes_ids(), list(
        np.array(s_graph.get_nodes_ids()) + len(a_graph_nodes_ids) + 1)))
    # a_graph.name = "A-Graph"
    # for node_id in list(a_graph.get_nodes_ids()):
    #     for attr_key in copy.deepcopy(a_graph.get_attributes_of_node(node_id)):
    #         if attr_key not in ["type", "Geometric_info"]:
    #             del a_graph.graph.nodes[node_id][attr_key]

    # s_graph.name = "S-Graph"
    # for node_id in list(s_graph.get_nodes_ids()):
    #     for attr_key in copy.deepcopy(s_graph.get_attributes_of_node(node_id)):
    #         if attr_key not in ["type", "Geometric_info"]:
    #             del s_graph.graph.nodes[node_id][attr_key]
    s_graph.relabel_nodes(mapping, copy=True)
    s_graph.stringify_node_ids()
    as_graph = copy.deepcopy(a_graph)
    as_graph.add_nodes(s_graph.get_attributes_of_all_nodes())
    as_graph.add_edges(s_graph.get_attributes_of_all_edges())
    as_dataset.append(as_graph)

    visualize_nxgraph_3d(
        as_dataset[0], "as_graph", visualize_alone=True, include_node_ids=True)

    # CREATE GRAPH MATCHER

    graph_matcher = GraphMatcher(fake_logger, log_level=0)
    graph_matcher.set_parameters(syntheticDS_params)
    a_nodes = a_graph.get_attributes_of_all_nodes()
    s_nodes = s_graph.get_attributes_of_all_nodes()
    print("********************************************************")
    print("a_graph nodes:")
    print("********************************************************")
    for node in a_nodes:
        print(node)
    print("********************************************************")
    print("s_graph nodes:")
    print("********************************************************")
    for node in s_nodes:
        print(node)
    print("********************************************************")

    graph_matcher.set_graph_from_wrapper(a_graph, "A-Graph")
    graph_matcher.set_graph_from_wrapper(s_graph, "S-Graph")

    # visualize_nxgraph(as_graph, "as_graph")

    # MATCH
    # time.sleep(99)
    success, matches, matches_full, matches_dev = graph_matcher.match(
        "A-Graph", "S-Graph")

    print(f"dbg {success=}")
    print(f"dbg {len(matches)=}")
    print(f"dbg {len(matches_full)=}")
    print(f"dbg {matches_dev=}")

    # wait for user input to continue
    input("Press Enter to continue...")

    # Close the graph visualizer window
    plt.close('all')

    # for final_combination in matches_full:
    #     print(f"flag new final combination")
    #     for i in final_combination:
    #         print(f"flag {i['origin_node_attrs']['type']} {i['score']}")

    if success:
        if len(matches_full) > 0:
            edges = []
            for edge_dict in matches_full[0]:
                edges.append((str(edge_dict['origin_node']), str(
                    edge_dict['target_node']), {"viz_feat": 'green'}))
            as_graph.add_edges(edges)
            visualize_nxgraph_3d(
                as_dataset[0], "as_graph", visualize_alone=True, include_node_ids=True)

        if len(matches) > 0:
            print(f"dbg {len(matches)=}")
            for i, match in enumerate(matches):
                edges = []
                for edge_dict in match:
                    edges.append((str(edge_dict['origin_node']),
                                  str(edge_dict['target_node']), {"viz_feat": 'red'}))
                    aux_as_graph = copy.deepcopy(as_graph)
                    aux_as_graph.add_edges(edges)
                visualize_nxgraph_3d(aux_as_graph, f"as_graph_match_{i}",
                                     visualize_alone=True, include_node_ids=True)

