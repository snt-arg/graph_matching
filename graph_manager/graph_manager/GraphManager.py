import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import time
import copy
import collections
import open3d as o3d
import  clipperpy


class GraphManager():
    def __init__(self) -> None:
        self.graphs = {}

    # def matchLegacyCustom(self, G1_name, G2_name):
    #     start_time = time.time()
    #     type_matches = self.matchByNodeType(G1_name, G2_name)
    #     print("Found {} candidates after isomorphism and cathegorical in type matching".format(len(type_matches)))
    #     good_matches = []
    #     type_match_candidates = copy.deepcopy(type_matches)
    #     condition = False
    #     while len(type_match_candidates) != 0:
    #         j = np.random.randint(0, len(type_match_candidates))
    #         type_match = type_match_candidates[j]
    #         type_match_candidates.pop(j)

    #         subgraph = self.graphs[G1_name].subgraph(list(type_match.keys()))
    #         # plot_options = {'node_size': 50, 'width': 2, 'with_labels' : True}
    #         # self.drawAnyGraph("trial", subgraph, plot_options, True)
    #         room_nodes = [node for node,attrs in subgraph.nodes(data=True) if attrs['type']=='room']
    #         condition = True
    #         for room_node in room_nodes:
    #             # print("flag")
    #             wall_nodes = [node for node in subgraph.neighbors(room_node) if subgraph.nodes(data=True)[node]['type']=='wall']
    #             submatch = {key: type_match[key] for key in wall_nodes}
    #             if not self.checkWallsGeometry(subgraph, self.graphs[G2_name], submatch):
    #                 condition = False
    #                 wrong_room_match = dict((k, type_match[k]) for k in wall_nodes)
    #                 for i, candidate in enumerate(type_match_candidates):
    #                     if len({k: wrong_room_match[k] for k in wrong_room_match if k in candidate and wrong_room_match[k] == candidate[k]}) == len(wrong_room_match):
    #                         type_match_candidates.pop(i)
    #                 break

    #         if condition:
    #             print("Geometric match found!")
    #             good_matches += [type_match]

    #     print(good_matches)
    #     print("Elapsed time in custom matching: {}".format(time.time() - start_time))
    #     return(condition, good_matches)


    def matchCustom(self, G1_name, G2_name):
        start_time = time.time()
        sweeped_levels = ["floor", "room", "room"]

        ## Room level
        lvl = 2
        G1_up_down = self.filter_graph_by_node_types(self.graphs[G1_name], sweeped_levels[:lvl])
        G2_up_down = self.filter_graph_by_node_types(self.graphs[G2_name], sweeped_levels[:lvl])
        type_matches = self.matchByNodeType(G1_up_down, G2_up_down)
        print("Found {} candidates after isomorphism and cathegorical in type matching".format(len(type_matches)))
        A_categorical, C = self.matches_to_level_association_matrix(G1_up_down, type_matches, sweeped_levels[lvl])
        data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_up_down, G2_up_down, A_categorical, "pos")
        clipper_match_numerical = self.compute_clipper(data1, data2, A_numerical)
        clipper_match_categorical = self.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)

        ## Wall level
        for lowerlevel_match in clipper_match_categorical:


    ## Graph-wise functions

    def setGraph(self, graph_def):
        graph = nx.Graph()
        graph.add_nodes_from(graph_def['nodes'])
        graph.add_edges_from(graph_def['edges'])
        self.graphs[graph_def['name']] = graph


    def categoricalMatch(self, G1, G2, categorical_condition, draw = False):
        GM = isomorphism.GraphMatcher(G1, G2, node_match=categorical_condition)
        matches = []
        if GM.subgraph_is_isomorphic():
            for subgraph in GM.subgraph_isomorphisms_iter():
                matches.append(subgraph)
                if draw:
                    plot_options = {
                        'node_size': 50,
                        'width': 2,
                        'with_labels' : True}
                    plot_options = self.defineColorPlotOptionFromMatch(G1, plot_options, subgraph, "blue", "red")
                    self.drawAnyGraph("Graph", G1, None, True)

        return matches ### TODO What should be returned?


    def matchByNodeType(self, G1, G2, draw= False):
        categorical_condition = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G1, G2, categorical_condition, draw)
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


    def filter_graph_by_node_types(self, graph, types):
        def filer_node_fn(node):
            return True if graph.nodes(data=True)[node]["type"] in types else False 

        graph_filtered = nx.subgraph_view(graph, filter_node=filer_node_fn)
        return graph_filtered

    
    def matches_to_level_association_matrix(self, origin_graph, matches, level):
        matches_as_tuple = [list(zip(match.keys(), match.values())) for match in matches]
        A = set()
        [A.update(match) for match in matches_as_tuple]
        A = [x for x in A if origin_graph.nodes(data=True)[x[0]]["type"] == level]
        C = np.eye((len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                if any(all(x in sublist for x in [A[i], A[j]]) for sublist in matches_as_tuple):
                    C[i,j] = C[j,i] = 1
        A_matrix = np.array(A)
        return A_matrix, C

    def generate_clipper_input(self, G1, G2, A_categorical, feature_name):
        nodes1, nodes2 = list(set(A_categorical[:,0])), list(set(A_categorical[:,1]))
        data1 = self.stack_nodes_feature(G1, nodes1, feature_name)
        data2 = self.stack_nodes_feature(G2, nodes2, feature_name)
        A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
        return(data1, data2, A_numerical, nodes1, nodes2)

    def stack_nodes_feature(self, graph, node_list, feature):
        return np.array([graph.nodes(data=True)[key][feature] for key in node_list])
        

    ## Geometry functions

    def planeIntersection(self, plane_1, plane_2, plane_3):
        normals = np.array([plane_1[:3],plane_2[:3],plane_3[:3]])
        # normals = np.array([[1,0,0],[0,1,0],[0,0,1]])
        distances = -np.array([plane_1[3],plane_2[3],plane_3[3]])
        # distances = -np.array([5,7,9])
        return(np.linalg.inv(normals).dot(distances.transpose()))


    def computePlanesSimilarity(self, walls_1_translated, walls_2_translated, thresholds = [0.001,0.001,0.001,0.001]):
        differences = walls_1_translated - walls_2_translated
        conditions = differences < thresholds
        asdf

    
    def computePoseSimilarity(self, rooms_1_translated, rooms_2_translated, thresholds = [0.001,0.001,0.001,0.001,0.001,0.001]):
        differences = rooms_1_translated - rooms_2_translated
        conditions = differences < thresholds
        asdf


    def computeWallsConsistencyMatrix(self, ):
        pass


    def changeWallsOrigin(self, original, main1_wall_i, main2_wall_i):
        # start_time = time.time()
        original[:,2] = np.array(np.zeros(original.shape[0]))   ### 2D simplification
        normalized = original / np.sqrt(np.power(original[:,:-1],2).sum(axis=1))[:, np.newaxis]
        intersection_point = self.planeIntersection(normalized[main1_wall_i,:], normalized[main2_wall_i,:], np.array([0,0,1,0]))

        #### Compute rotation for new origin
        z_normal = np.array([0,0,1]) ### 2D simplification
        x_axis_new_origin = np.cross(normalized[main1_wall_i,:3], z_normal)
        rotation = np.array((x_axis_new_origin,normalized[main1_wall_i,:3], z_normal))

        #### Build transform matrix
        rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
        translation_1 = np.array([np.concatenate((intersection_point, np.array([1.0])), axis=0)])
        full_transformation_matrix = np.concatenate((rotation_0, -translation_1), axis=0)

        #### Matrix multiplication
        transformed = np.transpose(np.matmul(full_transformation_matrix,np.matrix(np.transpose(original))))
        transformed_normalized = transformed / np.sqrt(np.power(transformed[:,:-1],2).sum(axis=1))
        # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
        return transformed_normalized


    def checkWallsGeometry(self, graph_1, graph_2, match):
        start_time = time.time()
        match_keys = list(match.keys())
        room_1 = np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])
        room_2 = np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])
        room_1_translated = self.changeWallsOrigin(room_1,0,1)
        room_2_translated = self.changeWallsOrigin(room_2,0,1)
        scores = self.computePlanesSimilarity(room_1_translated,room_2_translated)
        return False


    ## Clipper functions

    def compute_clipper(self, D1, D2, A):
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = 0.015
        iparams.epsilon = 0.02
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)

        t0 = time.perf_counter()
        clipper.score_pairwise_consistency(D1.T, D2.T, A)
        C = clipper.get_constraint_matrix()
        # print("A - Association matrix\n", A)
        # print("C - Constraint matrix\n", C)
        M = clipper.get_affinity_matrix()
        # print("M - Affinity matrix\n",M)
        t1 = time.perf_counter()
        print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

        t0 = time.perf_counter()
        clipper.solve()
        t1 = time.perf_counter()

        # A = clipper.get_initial_associations()
        Ain = clipper.get_selected_associations()
        print("Ain - Affinity matrix\n",Ain)
        return Ain

    def categorize_clipper_output(self, Ain_numerical, nodes1, nodes2):
        Ain_categorical = np.array([[nodes1[pair[0]],nodes2[pair[1]]] for pair in Ain_numerical])
