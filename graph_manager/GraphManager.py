import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt


SCORE_THR = 0.9


class GraphManager():
    def __init__(self, graph_def = None, graph_obj = None):
        if graph_def:
            graph = nx.Graph()
            graph.add_nodes_from(graph_def['nodes'])
            graph.add_edges_from(graph_def['edges'])
            self.name = graph_def['name']
            self.graph = graph

        elif graph_obj:
            self.name = "unknown"
            self.graph = graph_obj
        else:
            print("GraphManager: definition of graph not provided")


    def get_graph(self):
        return(self.graph)


    def categoricalMatch(self, G2, categorical_condition, draw = False):
        graph_matcher = isomorphism.GraphMatcher(self.graph, G2.get_graph(), node_match=categorical_condition, edge_match = lambda *_: True)
        matches = []
        if graph_matcher.subgraph_is_isomorphic():
            for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                matches.append(subgraph)
                if draw:
                    plot_options = {
                        'node_size': 50,
                        'width': 2,
                        'with_labels' : True}
                    plot_options = self.defineColorPlotOptionFromMatch(self.graph, plot_options, subgraph, "blue", "red")
                    self.draw("Graph {}".format(self.name), None, True)

        return matches ### TODO What should be returned?


    def matchByNodeType(self, G2, draw= False):
        categorical_condition = isomorphism.categorical_node_match(["type"], ["none"])
        matches = self.categoricalMatch(G2, categorical_condition, draw)
        matches_as_tuple = [list(zip(match.keys(), match.values())) for match in matches]
        # matches_as_list = [[[key, match[key]] for key in match.keys()] for match in matches]
        print("GM: Found {} candidates after isomorphism and cathegorical in type matching".format(len(matches_as_tuple),))
        return matches_as_tuple


    # def matchIsomorphism(self, G1_name, G2_name):
    #     return isomorphism.GraphMatcher(self.graphs[G1_name], self.graphs[G2_name])


    def draw(self, fig_name, options = None, show = False):
        if not options:
            options = {'node_color': 'red', 'node_size': 50, 'width': 2, 'with_labels' : True}

        fig = plt.figure(fig_name)
        ax = plt.gca()
        ax.clear()
        nx.draw(self.graph, **options)
        if show:
            plt.show(block=False)
            plt.pause(0.001)

    
    def defineColorPlotOptionFromMatch(self, graph, options, subgraph, old_color, new_color):
        colors = dict(zip(graph.nodes(), [old_color] * len(graph.nodes())))
        for origin_node in subgraph:
            colors[origin_node] = new_color
        options['node_color'] = colors.values()
        return options


    def filter_graph_by_node_types(self, types):
        def filter_node_fn(node):
            return True if self.graph.nodes(data=True)[node]["type"] in types else False 

        graph_filtered = GraphManager(graph_obj = nx.subgraph_view(self.graph, filter_node=filter_node_fn))
        return graph_filtered


    def stack_nodes_feature(self, node_list, feature):
        return np.array([self.graph.nodes(data=True)[key][feature] for key in node_list]).astype(np.float64)


    def get_neighbourhood_graph(self, node_name):
        neighbours = self.graph.neighbors(node_name)
        filtered_neighbours_names = list([n for n in neighbours]) + [node_name]
        subgraph = GraphManager(graph_obj= self.graph.subgraph(filtered_neighbours_names))
        return(subgraph)


    def get_total_number_nodes(self):
        return(len(self.graph.nodes(data=True)))


    def add_subgraph(self, nodes_def, edges_def):
        [self.graph.add_node(node_def[0], **node_def[1]) for node_def in nodes_def]
        [self.graph.add_edge(edge_def[0], edge_def[1]) for edge_def in edges_def]


    def set_draw_color_option_by_node_type(self, ):
        color_pallette = ["blue", "red", "orange", "cyan", "yellow"]
        type_list = [node[1]["type"] for node in self.graph.nodes(data=True)]
        type_set = list(set(type_list))
        colors = [color_pallette[type_set.index(node_type)] for node_type in type_list]

        # colors = dict(zip(graph.nodes(), [old_color] * len(graph.nodes())))
        # for origin_node in subgraph:
        #     colors[origin_node] = new_color
        # options['node_color'] = colors.values()
        return colors


    # ## Geometry functions

    # def planeIntersection(self, plane_1, plane_2, plane_3):
    #     normals = np.array([plane_1[:3],plane_2[:3],plane_3[:3]])
    #     # normals = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #     distances = -np.array([plane_1[3],plane_2[3],plane_3[3]])
    #     # distances = -np.array([5,7,9])
    #     return(np.linalg.inv(normals).dot(distances.transpose()))


    # def computePlanesSimilarity(self, walls_1_translated, walls_2_translated, thresholds = [0.001,0.001,0.001,0.001]):
    #     differences = walls_1_translated - walls_2_translated
    #     conditions = differences < thresholds
    #     asdf

    
    # def computePoseSimilarity(self, rooms_1_translated, rooms_2_translated, thresholds = [0.001,0.001,0.001,0.001,0.001,0.001]):
    #     differences = rooms_1_translated - rooms_2_translated
    #     conditions = differences < thresholds
    #     asdf


    # def computeWallsConsistencyMatrix(self, ):
    #     pass


    # def changeWallsOrigin(self, original, main1_wall_i, main2_wall_i):
    #     # start_time = time.time()
    #     original[:,2] = np.array(np.zeros(original.shape[0]))   ### 2D simplification
    #     normalized = original / np.sqrt(np.power(original[:,:-1],2).sum(axis=1))[:, np.newaxis]
    #     intersection_point = self.planeIntersection(normalized[main1_wall_i,:], normalized[main2_wall_i,:], np.array([0,0,1,0]))

    #     #### Compute rotation for new origin
    #     z_normal = np.array([0,0,1]) ### 2D simplification
    #     x_axis_new_origin = np.cross(normalized[main1_wall_i,:3], z_normal)
    #     rotation = np.array((x_axis_new_origin,normalized[main1_wall_i,:3], z_normal))

    #     #### Build transform matrix
    #     rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
    #     translation_1 = np.array([np.concatenate((intersection_point, np.array([1.0])), axis=0)])
    #     full_transformation_matrix = np.concatenate((rotation_0, -translation_1), axis=0)

    #     #### Matrix multiplication
    #     transformed = np.transpose(np.matmul(full_transformation_matrix,np.matrix(np.transpose(original))))
    #     transformed_normalized = transformed / np.sqrt(np.power(transformed[:,:-1],2).sum(axis=1))
    #     # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
    #     return transformed_normalized


    # def checkWallsGeometry(self, graph_1, graph_2, match):
    #     start_time = time.time()
    #     match_keys = list(match.keys())
    #     room_1 = np.array([graph_1.nodes(data=True)[key]["pos"] for key in match_keys])
    #     room_2 = np.array([graph_2.nodes(data=True)[match[key]]["pos"] for key in match_keys])
    #     room_1_translated = self.changeWallsOrigin(room_1,0,1)
    #     room_2_translated = self.changeWallsOrigin(room_2,0,1)
    #     scores = self.computePlanesSimilarity(room_1_translated,room_2_translated)
    #     return False


