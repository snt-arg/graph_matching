#!/usr/bin/python3
# EASY-INSTALL-ENTRY-SCRIPT: 'venv==0.0.0','console_scripts','venv'
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import time
import copy
import collections

class GraphManager():
    def __init__(self) -> None:
        self.graphs = {}

    def setGraph(self, graph_def):
        graph = nx.Graph()
        graph.add_nodes_from(graph_def['nodes'])
        graph.add_edges_from(graph_def['edges'])
        self.graphs[graph_def['name']] = graph

    def categoricalMatch(self, G1_name, G2_name, categorical_condition, draw = False):
        GM = isomorphism.GraphMatcher(self.graphs[G1_name], self.graphs[G2_name], node_match=categorical_condition)
        matches = []
        if GM.subgraph_is_isomorphic():
            print("graph")
            for subgraph in GM.subgraph_isomorphisms_iter():
                matches.append(subgraph)
                if draw:
                    plot_options = {
                        'node_size': 50,
                        'width': 2,
                        'with_labels' : True}
                    plot_options = self.defineColorPlotOptionFromMatch(self.graphs[G1_name], plot_options, subgraph, "blue", "red")
                    self.plotGraphByName(G1_name, plot_options)

        return matches ### TODO What should be returned?

    def matchByNodeType(self, G1_name, G2_name, draw= False):
        categorical_condition = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G1_name, G2_name, categorical_condition, draw)
        print("categorical", len(matches))
        return matches

    def matchIsomorphism(self, G1_name, G2_name):
        return isomorphism.GraphMatcher(self.graphs[G1_name], self.graphs[G2_name])

    def plotGraphByName(self, name, options = None, show = False):
        self.drawAnyGraph("Graph {}".format(name), self.graphs[name], options, True)


    def drawAnyGraph(self, fig_name, graph, options = None, show = False):
        if not options:
            options = {'node_color': 'red', 'node_size': 50, 'width': 2, 'with_labels' : True}

        fig = plt.figure(fig_name)
        nx.draw(graph, **options)
        if show:
            plt.show()

    
    def defineColorPlotOptionFromMatch(self, graph, options, subgraph, old_color, new_color):
        colors = dict(zip(graph.nodes(), [old_color] * len(graph.nodes())))
        for origin_node in subgraph:
            colors[origin_node] = new_color
        options['node_color'] = colors.values()
        return options

    def matchCustom(self, G1_name, G2_name):
        start_time = time.time()
        type_matches = self.matchByNodeType(G1_name, G2_name)
        print("Found {} candidates after isomorphism and cathegorical in type matching".format(len(type_matches)))
        good_matches = []
        type_match_candidates = copy.deepcopy(type_matches)
        condition = False
        while len(type_match_candidates) != 0:
            j = np.random.randint(0, len(type_match_candidates))
            type_match = type_match_candidates[j]
            type_match_candidates.pop(j)

            subgraph = self.graphs[G1_name].subgraph(list(type_match.keys()))
            # plot_options = {'node_size': 50, 'width': 2, 'with_labels' : True}
            # self.drawAnyGraph("trial", subgraph, plot_options, True)
            room_nodes = [node for node,attrs in subgraph.nodes(data=True) if attrs['type']=='room']
            condition = True
            for room_node in room_nodes:
                # print("flag")
                wall_nodes = [node for node in subgraph.neighbors(room_node) if subgraph.nodes(data=True)[node]['type']=='wall']
                submatch = {key: type_match[key] for key in wall_nodes}
                if not self.checkWallsGeometry(subgraph, self.graphs[G2_name], submatch):
                    condition = False
                    wrong_room_match = dict((k, type_match[k]) for k in wall_nodes)
                    for i, candidate in enumerate(type_match_candidates):
                        if len({k: wrong_room_match[k] for k in wrong_room_match if k in candidate and wrong_room_match[k] == candidate[k]}) == len(wrong_room_match):
                            type_match_candidates.pop(i)
                    break

            if condition:
                print("Geometric match found!")
                good_matches += [type_match]

        print(good_matches)
        print("Elapsed time in custom matching: {}".format(time.time() - start_time))
        return(condition, good_matches)


    def checkWallsGeometry(self, graph_1, graph_2, match):
        start_time = time.time()
        match_keys = list(match.keys())
        room_1 = np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])
        room_2 = np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])
        room_1_transformed = self.computeRoomTransformedData(room_1)
        room_2_transformed = self.computeRoomTransformedData(room_2)
        return False


    def computeRoomTransformedData(self, original):
        # start_time = time.time()
        original[:,2] = np.array(np.zeros(original.shape[0]))   ### 2D simplification
        normalized = original / np.sqrt(np.power(original[:,:-1],2).sum(axis=1))[:, np.newaxis]
        intersection_point = self.planeIntersection(normalized[0,:], normalized[1,:], np.array([0,0,1,0]))

        #### Compute rotation for new origin
        z_normal = np.array([0,0,1]) ### 2D simplification
        x_axis_new_origin = np.cross(normalized[0,:3], z_normal)
        rotation = np.array((x_axis_new_origin,normalized[0,:3], z_normal))

        #### Build transform matrix
        rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
        translation_1 = np.array([np.concatenate((intersection_point, np.array([1.0])), axis=0)])
        full_transformation_matrix = np.concatenate((rotation_0, -translation_1), axis=0)

        #### Matrix multiplication
        transformed = np.transpose(np.matmul(full_transformation_matrix,np.matrix(np.transpose(original))))
        transformed_normalized = transformed / np.sqrt(np.power(transformed[:,:-1],2).sum(axis=1))
        # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
        return transformed_normalized


    def planeIntersection(self, plane_1, plane_2, plane_3):
        normals = np.array([plane_1[:3],plane_2[:3],plane_3[:3]])
        # normals = np.array([[1,0,0],[0,1,0],[0,0,1]])
        distances = -np.array([plane_1[3],plane_2[3],plane_3[3]])
        # distances = -np.array([5,7,9])
        return(np.linalg.inv(normals).dot(distances.transpose()))
