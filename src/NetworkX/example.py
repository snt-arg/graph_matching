from re import sub
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt

fig1 = plt.figure("Original graphs to match")

### Definition of S_Graph from BIM information
G_BIM = nx.Graph()
bim_nodes_floors_attrs = [("floor_1", {"type": "floor", "pos": [0,0]})]
G_BIM.add_nodes_from(bim_nodes_floors_attrs)
bim_nodes_rooms_attrs = [("room_1", {"type": "room", "pos": [0,0]}), ("room_2", {"type": "room", "pos": [5,0]}), ("room_3", {"type": "room", "pos": [10,0]})]
G_BIM.add_nodes_from(bim_nodes_rooms_attrs)
bim_nodes_walls_attrs = [("wall_1", {"type": "wall", "pos": [-2,2]}), ("wall_2", {"type": "wall", "pos": [2,-2]}),("wall_3", {"type": "wall", "pos": [2,2]}),\
                        ("wall_4", {"type": "wall", "pos": [-2,-2]}),("wall_5", {"type": "wall", "pos": [3,2]}), ("wall_6", {"type": "wall", "pos": [3,-2]}),\
                        ("wall_7", {"type": "wall", "pos": [7,2]}), ("wall_8", {"type": "wall", "pos": [7,-2]}),("wall_9", {"type": "wall", "pos": [8,2]}),\
                        ("wall_10",{"type": "wall", "pos": [8,-2]}),("wall_11", {"type": "wall", "pos": [12,2]}), ("wall_12", {"type": "wall", "pos": [12,-2]})]
G_BIM.add_nodes_from(bim_nodes_walls_attrs)

bim_edges_floors_attrs = [("room_1","floor_1"),("room_2","floor_1"),("room_3","floor_1")]
G_BIM.add_edges_from(bim_edges_floors_attrs)
bim_edges_rooms_attrs = [("room_1","wall_1"),("room_1","wall_2"),("room_1","wall_3"), ("room_1","wall_4"),("room_2","wall_5"),\
    ("room_2","wall_6"),("room_2","wall_7"), ("room_2","wall_8"),("room_3","wall_9"),("room_3","wall_10"),("room_3","wall_11"),\
    ("room_3","wall_12")]
G_BIM.add_edges_from(bim_edges_rooms_attrs)

bim_plot_options = {
    'node_color': 'blue',
    'node_size': 50,
    'width': 2,
    'with_labels' : True,
}
subax1 = plt.subplot(121)
nx.draw(G_BIM, **bim_plot_options)


### Definition of S_Graph from real robot information
G_REAL = nx.Graph()
real_nodes_rooms_attrs = [("room_1", {"type": "room", "pos": [5,0]})]
# real_nodes_rooms_attrs = [("room_1", {"type": "room", "pos": [10,0]})]
G_REAL.add_nodes_from(real_nodes_rooms_attrs)
real_nodes_walls_attrs = [("wall_1", {"type": "wall", "pos": [3,2]}), ("wall_2", {"type": "wall", "pos": [7,2]}),("wall_3", {"type": "wall", "pos": [3,-2]})]
# real_nodes_walls_attrs = [("wall_1", {"type": "wall", "pos": [8,2]}), ("wall_2", {"type": "wall", "pos": [12,2]}),("wall_3", {"type": "wall", "pos": [8,-2]})]

G_REAL.add_nodes_from(real_nodes_walls_attrs)

real_edges_rooms_attrs = [("room_1","wall_1"),("room_1","wall_2"),("room_1","wall_3")]
G_REAL.add_edges_from(real_edges_rooms_attrs)

real_plot_options = {
    'node_color': 'red',
    'node_size': 50,
    'width': 2,
    'with_labels' : True,
}
subax2 = plt.subplot(122)
nx.draw(G_REAL, **real_plot_options)


### Subgraph isomorphism matching
GM = isomorphism.GraphMatcher(G_BIM, G_REAL)
isomorphism.GraphMatcher
# print(GM.subgraph_is_isomorphic())
# print(GM.mapping)
# for isomorphism in GM.subgraph_isomorphisms_iter():
#     print(isomorphism)

### Categorical node match
nm_type_only = isomorphism.categorical_node_match(["type",], ["none"])
nm_with_pose = isomorphism.categorical_node_match(["type", "pos"], ["none", 99999999999])
# print(nm(G_BIM.nodes["wall_1"], G_REAL.nodes["wall_1"]))


### Subgraph isomorphism with categorical matching

GM = isomorphism.GraphMatcher(G_BIM, G_REAL, node_match=nm_with_pose)
matching_fig = plt.figure("Matching")
# ax = matching_fig.add_subplot(111)
if GM.subgraph_is_isomorphic():
    matches = []
    for subgraph in GM.subgraph_isomorphisms_iter():
        matches.append(subgraph)

        colors = dict(zip(G_BIM.nodes(), ["blue"] * len(G_BIM.nodes())))
        for origin_node in subgraph:
            colors[origin_node] = "red"        
        plot_options = {
            'node_color': colors.values(),
            'node_size': 50,
            'width': 2,
            'with_labels' : True}
        
        # ax.plot([1,2,3])
        nx.draw(G_BIM, **plot_options)

    print("Number of matches found:", len(matches))

plt.show()