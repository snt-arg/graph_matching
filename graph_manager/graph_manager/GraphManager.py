import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import time
import copy
import collections
import open3d as o3d
from Clipper import Clipper


class GraphManager():
    def __init__(self) -> None:
        self.graphs = {}


    def matchCustom(self, G1_name, G2_name):
        final_match = []
        sweeped_levels = ["floor", "room", "wall"]
        sweeped_levels_dt = ["points", "points", "points&normal"]
        full_graph_matches = self.matchByNodeType(self.graphs[G1_name], self.graphs[G2_name])

        ## Room level
        lvl = 1
        G1_to_rooms = self.filter_graph_by_node_types(self.graphs[G1_name], sweeped_levels[:lvl+1])
        G2_to_rooms = self.filter_graph_by_node_types(self.graphs[G2_name], sweeped_levels[:lvl+1])
        room_matches = self.matchByNodeType(G1_to_rooms, G2_to_rooms)
        # A_categorical = self.filter_match_by_node_type(G1_to_rooms, room_matches, sweeped_levels[lvl])
        A_categorical, _ = self.matches_to_level_association_matrix(G1_to_rooms, room_matches, [sweeped_levels[lvl]])
        data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_to_rooms, G2_to_rooms, A_categorical, "pos")
        clipper = Clipper(sweeped_levels_dt[lvl])
        clipper.score_pairwise_consistency(data1, data2, A_numerical)
        clipper_match_numerical, score = clipper.solve_clipper()
        clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)
        final_match.append(clipper_match_categorical)
        print("Found candidates for room matching: \n {}".format(clipper_match_categorical))
        
        ## Wall level
        rooms1_point_normal = self.change_pos_dt(G1_to_rooms, clipper_match_categorical[:,0], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
        rooms2_point_normal = self.change_pos_dt(G2_to_rooms, clipper_match_categorical[:,1], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
        lvl += 1
        G1_to_walls = self.filter_graph_by_node_types(self.graphs[G1_name], sweeped_levels[:lvl+1])
        G2_to_walls = self.filter_graph_by_node_types(self.graphs[G2_name], sweeped_levels[:lvl+1])
        for i, lowerlevel_match in enumerate(clipper_match_categorical):
            print("Checking room {}".format(lowerlevel_match))
            G1_room_neighborhood = self.get_neighbourhood_graph_by_type(G1_to_walls, lowerlevel_match[0])
            G2_room_neighborhood = self.get_neighbourhood_graph_by_type(G2_to_walls, lowerlevel_match[1])
            room_and_wall_matches = self.matchByNodeType(G1_room_neighborhood, G2_room_neighborhood)
            A_categorical, C_graph = self.matches_to_level_association_matrix(G1_room_neighborhood, room_and_wall_matches, sweeped_levels[lvl:lvl+1])
            A_categorical_and_parent_association = np.append(A_categorical, [lowerlevel_match], axis=0)
            G1_room_neighborhood.nodes(data=True)[lowerlevel_match[0]]["pos"] = rooms1_point_normal[i]
            G2_room_neighborhood.nodes(data=True)[lowerlevel_match[1]]["pos"] = rooms2_point_normal[i]
            data1, data2, A_numerical_and_parent_association, nodes1, nodes2 = self.generate_clipper_input(G1_room_neighborhood, G2_room_neighborhood, A_categorical_and_parent_association, "pos")
            clipper = Clipper(sweeped_levels_dt[lvl])
            clipper.score_pairwise_consistency(data1, data2, A_numerical_and_parent_association)
            M_aux, _ = clipper.get_M_C_matrices()
            clipper.remove_last_assoc_M_C_matrices()
            # clipper.filter_C_M_matrices(C_graph) # TODO: waiting for an answer in the issue
            # clipper.set_M_diagonal_values(M_aux[-1,:][:-1]) # TODO: waiting for an answer in the issue
            clipper_match_numerical, score = clipper.solve_clipper()
            clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)
            final_match.append(clipper_match_categorical)
            print("Found candidates for room {} matching: \n {}".format(i, clipper_match_categorical))

        return final_match
            

    ## Graph-wise functions

    def setGraph(self, graph_def):
        graph = nx.Graph()
        graph.add_nodes_from(graph_def['nodes'])
        graph.add_edges_from(graph_def['edges'])
        self.graphs[graph_def['name']] = graph


    def categoricalMatch(self, G1, G2, categorical_condition, draw = False):
        graph_matcher = isomorphism.GraphMatcher(G1, G2, node_match=categorical_condition, edge_match = lambda *_: True)
        matches = []
        if graph_matcher.subgraph_is_isomorphic():
            for subgraph in graph_matcher.subgraph_isomorphisms_iter():
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
        matches_as_tuple = [list(zip(match.keys(), match.values())) for match in matches]
        print("GM: Found {} candidates after isomorphism and cathegorical in type matching".format(len(matches_as_tuple),))
        # print(matches_as_tuple)
        return matches_as_tuple


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

    
    def matches_to_level_association_matrix(self, origin_graph, matches, levels):
        A = set()
        [A.update(match) for match in matches]
        A = [x for x in A if origin_graph.nodes(data=True)[x[0]]["type"] in levels]
        C = np.eye((len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                if any(all(x in sublist for x in [A[i], A[j]]) for sublist in matches):
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
        return np.array([graph.nodes(data=True)[key][feature] for key in node_list]).astype(np.float64)


    def get_neighbourhood_graph_by_type(self, graph, node_name):
        all_nodes = graph.nodes(data=True)
        neighbours = graph.neighbors(node_name)
        filtered_neighbours_names = list([n for n in neighbours]) + [node_name]
        subgraph = graph.subgraph(filtered_neighbours_names)
        return(subgraph)

    
    # def filter_match_by_node_type(self, origin_graph, matches, node_type):
    #     A = set()
    #     [A.update(match) for match in matches]
    #     A = [x for x in A if origin_graph.nodes(data=True)[x[0]]["type"] == node_type]
    #     return np.array(A)
        

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


    def change_pos_dt(self, graph, match, in_dt, out_dt):
        original = self.stack_nodes_feature(graph, match, "pos")

        if in_dt == out_dt:
            processed = original
        elif in_dt == "points" and out_dt == "points&normal":
            diffs = original - np.concatenate((original[1:], [original[0]]),axis=0)
            normal = diffs / np.sqrt((diffs**2).sum())
            processed = np.concatenate((original, normal),axis=1)

        return(processed)