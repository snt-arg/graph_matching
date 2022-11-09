import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import sympy as sp
import time
import copy
import collections

class GraphManager():
    def __init__(self) -> None:
        self.graphs = {}

    def setGraph(self, name, node_attrs, edge_attrs):
        graph = nx.Graph()
        graph.add_nodes_from(node_attrs)
        graph.add_edges_from(edge_attrs)
        self.graphs[name] = graph

    def categoricalMatch(self, G1_name, G2_name, categorical_condition, draw = False):
        GM = isomorphism.GraphMatcher(self.graphs[G1_name], self.graphs[G2_name], node_match=categorical_condition)
        matches = []
        if GM.subgraph_is_isomorphic():
            for subgraph in GM.subgraph_isomorphisms_iter():
                matches.append(subgraph)
                if draw:
                    plot_options = {
                        'node_size': 50,
                        'width': 2,
                        'with_labels' : True}
                    plot_options = self.defineColorPlotOptionFromMatch(self.graphs[G1_name], plot_options, subgraph, "blue", "red")
                    self.plotGraphByName(G1_name, plot_options)

        return matches ### TODO What should be returned

    def matchByNodeType(self, G1_name, G2_name, draw= False):
        nm_with_pose = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G1_name, G2_name, nm_with_pose, draw)
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
        room_1 = {"raw":np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])}
        room_2 = {"raw":np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])}
        room_1["raw"][:,2] = np.array(np.zeros(room_1["raw"].shape[0]))   ### TODO 2D simplification
        room_2["raw"][:,2] = np.array(np.zeros(room_2["raw"].shape[0]))   ### TODO 2D simplification
        room_1["normalized"] = room_1["raw"] / np.sqrt(np.power(room_1["raw"][:,:-1],2).sum(axis=1))[:, np.newaxis]
        room_2["normalized"] = room_2["raw"] / np.sqrt(np.power(room_2["raw"][:,:-1],2).sum(axis=1))[:, np.newaxis]
        # print("Elapsed time in first ops: {}".format(time.time() - start_time))

        ### Norm condition
        room_1["relative_norm"] = room_1["normalized"][1:,:] - room_1["normalized"][0,:]
        room_2["relative_norm"] = room_2["normalized"][1:,:] - room_2["normalized"][0,:]
        difference_norms = room_1["relative_norm"] - room_2["relative_norm"]
        if not(abs(difference_norms).sum() < 0.001): #TODO assert
            return False
        # print("Elapsed time in norm ops: {}".format(time.time() - start_time))

        ### Distance condition in a 2D case, using 3D manifold
        room_1["intersection_point"] = self.planeIntersection(room_1["normalized"][0,:], room_1["normalized"][1,:], np.array([0,0,1,0]))
        room_2["intersection_point"] = self.planeIntersection(room_2["normalized"][0,:], room_2["normalized"][1,:], np.array([0,0,1,0]))
        # print("Elapsed time after intersection: {}".format(time.time() - start_time))
        
        #### Library mode
        # room_1["transformed_distances"] = [self.distancePlanePoint(room_1["normalized"][i,:], room_1["intersection_point"]) for i in np.arange(room_1["raw"].shape[0])[2:]]
        # room_2["transformed_distances"] = [self.distancePlanePoint(room_2["normalized"][i,:], room_2["intersection_point"]) for i in np.arange(room_2["raw"].shape[0])[2:]]
        #### Equations mode
        room_1["transformed_distances"] = self.transformedPlaneDistance(room_1["normalized"], room_1["intersection_point"])
        room_2["transformed_distances"] = self.transformedPlaneDistance(room_2["normalized"], room_2["intersection_point"])
        difference_cp_distances = np.array(room_1["transformed_distances"]) - np.array(room_2["transformed_distances"])
        if not(abs(difference_cp_distances).sum() < 0.1): #TODO assert
            return False

        # print("Elapsed time in successful geometry matching: {}".format(time.time() - start_time))
        return True


    def planeIntersection(self, plane_1, plane_2, plane_3):
        ### Library mode
        # pl1 = sp.Plane(-plane_1[3]*plane_1[:3], normal_vector=plane_1[:3])
        # pl2 = sp.Plane(-plane_2[3]*plane_2[:3], normal_vector=plane_2[:3])
        # pl3 = sp.Plane(-plane_3[3]*plane_3[:3], normal_vector=plane_3[:3])
        # r = pl1.intersection(pl2)
        # # print(r)
        # p = r[0].intersection(pl3)
        # return(np.array(p[0]))

        ### Equations mode
        normals = np.array([plane_1[:3],plane_2[:3],plane_3[:3]])
        # normals = np.array([[1,0,0],[0,1,0],[0,0,1]])
        distances = -np.array([plane_1[3],plane_2[3],plane_3[3]])
        # distances = -np.array([5,7,9])
        return(np.linalg.inv(normals).dot(distances.transpose()))

    def distancePlanePoint(self, plane, point):
        pl = sp.Plane(-plane[3]*plane[:3], normal_vector=plane[:3])
        p = sp.Point(point)
        return(pl.distance(p))

    
    def transformedPlaneDistance(self, planes, new_point):
        normals = planes[2:,:3]
        prior_distances = planes[2:,3]
        # new_points = np.tile(new_point, (1, planes.shape[0]-2))
        # print(new_points)
        transformed_distances = prior_distances + normals.dot(new_point)
        return transformed_distances