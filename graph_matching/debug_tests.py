from graph_wrapper.GraphWrapper import GraphWrapper
import matplotlib.pyplot as plt
import copy
import numpy as np


#########     SET INITIAL GRAPHS AND GT
Agraph_nodes = [(5,{"type": "Infinite Room"}), (8,{"type": "Infinite Room"}),\
                (1,{"type": "Plane"}), (2,{"type": "Plane"}),(3,{"type": "Plane"}), (4,{"type": "Plane"}),\
                (6,{"type": "Plane"}), (7,{"type": "Plane"})]
Agraph_nodes = [(str(attr[0]), attr[1]) for attr in Agraph_nodes]
Agraph_edges = [(5,1),(5,2),(5,3),(5,4),(8,6),(8,7),(8,3),(8,4)]
Agraph_edges = [(str(attr[0]), str(attr[1])) for attr in Agraph_edges]
Agraph_dict = {"name": "Sgraphs", "nodes": Agraph_nodes, "edges": Agraph_edges}
agraph = GraphWrapper(Agraph_dict)

Sgraph_nodes = [(105,{"type": "Infinite Room"}), (108,{"type": "Infinite Room"}),\
                (101,{"type": "Plane"}), (102,{"type": "Plane"}),(103,{"type": "Plane"}), (104,{"type": "Plane"}),\
                (106,{"type": "Plane"}), (107,{"type": "Plane"})]
Sgraph_nodes = [(str(attr[0]), attr[1]) for attr in Sgraph_nodes]
Sgraph_edges = [(105,101),(105,102),(105,103),(105,104),(108,106),(108,107),(108,103),(108,104)]
Sgraph_edges = [(str(attr[0]), str(attr[1])) for attr in Sgraph_edges]
Sgraph_dict = {"name": "Agraphs", "nodes": Sgraph_nodes, "edges": Sgraph_edges}
sgraph = GraphWrapper(Sgraph_dict)

merged_graph = copy.deepcopy(sgraph)
merged_graph = merged_graph.merge_graph(agraph)
GT = {"Infinite Room": [[8,108],[5,105]], "Plane": [[1,101],[2,102],[3,103],[3,103],[4,104],[4,104],[6,106],[7,107]]}
GT_as_edges = [(pair[0], pair[1], {}) for pair in GT["Infinite Room"]] + [(pair[0], pair[1], {}) for pair in GT["Plane"]]
merged_graph.add_edges(GT_as_edges)


########      UPDATE REPEATED NODE IDS
### A GRAPHS
agraph_planes_ids = copy.deepcopy(agraph.filter_graph_by_node_types(["Plane"]).get_nodes_ids())
a_graph_mapping = {}

for node_id in agraph_planes_ids:
    plane_connections = list(agraph.edges_of_node(node_id))

    for i in range(len(plane_connections) - 1):
        max_agraph_node_id = max([int(i) for i in agraph.get_nodes_ids()])
        new_node_id = str(max_agraph_node_id + 1)
        a_graph_mapping.update({node_id: new_node_id})
        sgraph_gt_plane_id = np.array(GT["Plane"])[np.array(GT["Plane"])[:,0] == int(node_id)][i][1]
        sgraph_gt_room_id = list(sgraph.get_neighbourhood_graph(str(sgraph_gt_plane_id)).filter_graph_by_node_types(["Infinite Room"]).get_nodes_ids())[0]
        agraph_gt_room_id = np.array(GT["Infinite Room"])[np.array(GT["Infinite Room"])[:,1] == int(sgraph_gt_room_id)][0][0]
        agraph.remove_edges([(str(node_id), str(agraph_gt_room_id))])
        agraph.add_nodes([(str(new_node_id), agraph.get_attributes_of_node(node_id))])
        agraph.add_edges([(str(new_node_id), str(agraph_gt_room_id), {})])

        GT["Plane"].remove([int(node_id), int(sgraph_gt_plane_id)])
        GT["Plane"] = GT["Plane"] + [(int(new_node_id), int(sgraph_gt_plane_id))]


### S GRAPHS
sgraph_planes_ids = copy.deepcopy(sgraph.filter_graph_by_node_types(["Plane"]).get_nodes_ids())
s_graph_mapping = {}

for node_id in sgraph_planes_ids:
    plane_connections = list(sgraph.edges_of_node(node_id))
    
    for i in range(len(plane_connections) - 1):
        max_sgraph_node_id = max([int(i) for i in sgraph.get_nodes_ids()])
        new_node_id = str(max_sgraph_node_id + 1)
        s_graph_mapping.update({node_id: new_node_id})
        agraph_gt_plane_id = np.array(GT["Plane"])[np.array(GT["Plane"])[:,1] == int(node_id)][i][0]
        agraph_gt_room_id = list(agraph.get_neighbourhood_graph(str(agraph_gt_plane_id)).filter_graph_by_node_types(["Infinite Room"]).get_nodes_ids())[0]
        sgraph_gt_room_id = np.array(GT["Infinite Room"])[np.array(GT["Infinite Room"])[:,0] == int(agraph_gt_room_id)][0][1]
        sgraph.remove_edges([(str(node_id), str(sgraph_gt_room_id))])
        sgraph.add_nodes([(str(new_node_id), sgraph.get_attributes_of_node(node_id))])
        sgraph.add_edges([(str(new_node_id), str(sgraph_gt_room_id), {})])

        GT["Plane"].remove([int(agraph_gt_plane_id), int(node_id)])
        GT["Plane"] = GT["Plane"] + [(int(agraph_gt_plane_id), int(new_node_id))]



### PRINT
merged_graph_final = copy.deepcopy(sgraph)
merged_graph_final = merged_graph_final.merge_graph(agraph)
GT_as_edges_final = [(str(pair[0]), str(pair[1]), {}) for pair in GT["Infinite Room"]] + [(str(pair[0]), str(pair[1]), {}) for pair in GT["Plane"]]
merged_graph_final.add_edges(GT_as_edges_final)
merged_graph_final.draw("merged_graph_final", options = None, show = True)

plt.show()