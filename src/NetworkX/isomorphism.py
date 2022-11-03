import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt

## Non-directed isomorphism verification
G1 = nx.path_graph(4)
G2 = nx.path_graph(4)
GM = isomorphism.GraphMatcher(G1, G2)
# print(GM.is_isomorphic())
# print(GM.mapping)

# subax1 = plt.subplot(131)
# nx.draw(G1, with_labels=True, font_weight='bold')
# subax2 = plt.subplot(132)
# nx.draw(G2, with_labels=True, font_weight='bold')
# plt.show()

## Directed isomorphism verification
G1 = nx.path_graph(4, create_using=nx.DiGraph())
G2 = nx.path_graph(4, create_using=nx.DiGraph())
DiGM = isomorphism.DiGraphMatcher(G1, G2)
# print(DiGM.is_isomorphic())
# print(DiGM.mapping)

# subax1 = plt.subplot(131)
# nx.draw(G1, with_labels=True, font_weight='bold')
# subax2 = plt.subplot(132)
# nx.draw(G2, with_labels=True, font_weight='bold')
# plt.show()


## Testing for subgraphs isomorphism(not in tutorial)
G1 = nx.path_graph(6)
G2 = nx.path_graph(4)
GM = isomorphism.GraphMatcher(G1, G2)
# print(GM.subgraph_is_isomorphic())
# print(GM.mapping)
# print(GM.subgraph_isomorphisms_iter())
# for isomorphism in GM.subgraph_isomorphisms_iter():
#     print(isomorphism)

# subax1 = plt.subplot(131)
# nx.draw(G1, with_labels=True, font_weight='bold')
# subax2 = plt.subplot(132)
# nx.draw(G2, with_labels=True, font_weight='bold')
# plt.show()


## Testing for semantic feasibility --- TODO Not understood
G1 = nx.path_graph(4)
G2 = nx.path_graph(4)
G1.nodes[0]["color"] = "red"
G2.nodes[0]["color"] = "green"
GM = isomorphism.GraphMatcher(G1, G2)
# print(GM.semantic_feasibility(G1.nodes[0],G2.nodes[0]))


## Test for categorical matching of nodes
nm = isomorphism.categorical_node_match("color", 1)
print(nm(G1.nodes[0], G2.nodes[0]))
nm = isomorphism.categorical_node_match(["color", "size"], ["red", 2])
print(nm)