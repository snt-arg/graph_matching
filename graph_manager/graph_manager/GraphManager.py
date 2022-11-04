import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import sympy as sp


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

    def matchByType(self, G1_name, G2_name, draw= False):
        nm_with_pose = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G1_name, G2_name, nm_with_pose, draw)
        return matches


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

    def findFirstPose(self):
        type_matches = self.matchByType("bim", "real")
        good_matches = []
        for type_match in type_matches:
            subgraph = self.graphs["bim"].subgraph(list(type_match.keys()))
            plot_options = {'node_size': 50, 'width': 2, 'with_labels' : True}
            # self.drawAnyGraph("trial", subgraph, plot_options, True)
            room_nodes = [node for node,attrs in subgraph.nodes(data=True) if attrs['type']=='room']
            condition = True
            for room_node in room_nodes:
                wall_nodes = [node for node in subgraph.neighbors(room_node) if subgraph.nodes(data=True)[node]['type']=='wall']
                submatch = {key: type_match[key] for key in wall_nodes}
                if not self.checkWallsGeometry(subgraph, self.graphs['real'], submatch):
                    condition = False

            if condition:
                print("Geometric match found!")
                good_matches += [type_match]

        print(good_matches)
        return(condition, good_matches)


    def checkWallsGeometry(self, graph_1, graph_2, match):
        match_keys = list(match.keys())
        room_1 = {"raw":np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])}
        room_2 = {"raw":np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])}
        room_1["raw"][:,2] = np.array(np.zeros(room_1["raw"].shape[0]))   ### TODO 2D simplification
        room_2["raw"][:,2] = np.array(np.zeros(room_2["raw"].shape[0]))   ### TODO 2D simplification
        room_1["normalized"] = room_1["raw"] / np.sqrt(np.power(room_1["raw"][:,:-1],2).sum(axis=1))[:, np.newaxis]
        room_2["normalized"] = room_2["raw"] / np.sqrt(np.power(room_2["raw"][:,:-1],2).sum(axis=1))[:, np.newaxis]

        ### Norm condition
        room_1["relative_norm"] = room_1["normalized"][1:,:] - room_1["normalized"][0,:]
        room_2["relative_norm"] = room_2["normalized"][1:,:] - room_2["normalized"][0,:]
        relative_norm_distances = room_1["relative_norm"] - room_2["relative_norm"]
        if not(abs(relative_norm_distances).sum() < 0.001): #TODO assert
            return False

        ### Distance condition in a 2D case, using 3D manifold
        room_1["interaction_point"] = self.planeIntersection(room_1["normalized"][0,:], room_1["normalized"][1,:], np.array([0,0,1,0]))
        room_2["interaction_point"] = self.planeIntersection(room_2["normalized"][0,:], room_2["normalized"][1,:], np.array([0,0,1,0]))
        room_1["cp_distances"] = [self.distancePlanePoint(room_1["normalized"][i,:], room_1["interaction_point"]) for i in np.arange(room_1["raw"].shape[0])[2:]]
        room_2["cp_distances"] = [self.distancePlanePoint(room_2["normalized"][i,:], room_2["interaction_point"]) for i in np.arange(room_2["raw"].shape[0])[2:]]
        relative_cp_distances = np.array(room_1["cp_distances"]) - np.array(room_2["cp_distances"])
        if not(abs(relative_cp_distances).sum() < 0.1): #TODO assert
            return False

        return True


    def planeIntersection(self, plane_1, plane_2, plane_3):
        pl1 = sp.Plane(-plane_1[3]*plane_1[:3], normal_vector=plane_1[:3])
        pl2 = sp.Plane(-plane_2[3]*plane_2[:3], normal_vector=plane_2[:3])
        pl3 = sp.Plane(-plane_3[3]*plane_3[:3], normal_vector=plane_3[:3])
        r = pl1.intersection(pl2)
        # print(r)
        p = r[0].intersection(pl3)
        # print(p)
        return(np.array(p[0]))

    def distancePlanePoint(self, plane, point):
        pl = sp.Plane(-plane[3]*plane[:3], normal_vector=plane[:3])
        p = sp.Point(point)
        return(pl.distance(p))