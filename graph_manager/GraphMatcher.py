import numpy as np
from .GraphManager import GraphManager
from .Clipper import Clipper
import matplotlib.pyplot as plt

SCORE_THR = 0.99


class GraphMatcher():
    def __init__(self, logger):
        self.graphs = {}
        self.graphs["match"] = GraphManager(graph_def={'name': "match",'nodes' : [], 'edges' : []})
        self.logger = logger


    def setGraph(self, graph_def):
        self.graphs[graph_def['name']] = GraphManager(graph_def = graph_def)


    def match_custom(self, G1_name, G2_name):
        sweeped_levels = ["floor", "Finite Room", "Plane"]
        sweeped_levels_dt = ["points", "points", "points&normal"]
        full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        lvl = 0

        def match_iteration(G1, G2, lvl, parents_data = None):
            G1_lvl = G1.filter_graph_by_node_types(sweeped_levels[lvl])
            G2_lvl = G2.filter_graph_by_node_types(sweeped_levels[lvl])
            matches = G1_lvl.matchByNodeType(G2_lvl)
            # A_categorical, _ = self.matches_to_suitable_A_C(G1_lvl, matches, full_graph_matches, [sweeped_levels[lvl]])
            matches = self.filter_local_match_with_global(matches, full_graph_matches)
            filter1_scores = []
            filter1_matches = []
            nodes_id = []
            for A_categorical in matches:
                print("Checking candidate \n{}".format(A_categorical))
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, A_categorical, "Geometric_info")
                # if parents_data:
                #     # print("flag")
                #     # print(data1, data2, A_numerical)
                #     self.add_parents_data(data1, data2, A_numerical, parents_data)
                #     # print(data1, data2, A_numerical)
                clipper = Clipper(sweeped_levels_dt[lvl])
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                # if parents_data:
                #     M_aux, _ = clipper.get_M_C_matrices()
                #     clipper.remove_last_assoc_M_C_matrices()
                #     # clipper.filter_C_M_matrices(C_graph) # TODO: waiting for an answer in the issue
                #     # clipper.set_M_diagonal_values(M_aux[-1,:][:-1]) # TODO: waiting for an answer in the issue
                clipper_match_numerical, score = clipper.solve_clipper()
                clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)

                print("Found candidates with score {} for matching: \n {}".format(score, clipper_match_categorical))

                if score > SCORE_THR and self.check_match_not_in_list(clipper_match_categorical, filter1_matches):
                    filter1_scores.append(score)
                    filter1_matches.append(clipper_match_categorical)
                    node_id = self.graphs["match"].get_total_number_nodes() + 1
                    nodes_id.append(node_id)
                    node_attr = [(node_id, {"type": sweeped_levels[lvl], "match": A_categorical})]
                    if parents_data:
                        edges_attr = [(node_id, parents_data["id"])]
                    else:
                        edges_attr = []
                    self.graphs["match"].add_subgraph(node_attr, edges_attr)

            sorted_matches_indexes = np.argsort(filter1_scores)[::-1]
            # sorted_scores = np.sort(filter1_scores)[::-1]
            # best_matches_indeces = np.where(np.array(filter1_scores) > SCORE_THR)[0]
            # best_scores = [filter1_scores[i] for i in best_matches_indeces]
            # best_submatches = [filter1_matches[i] for i in best_matches_indeces]
            # print(best_scores)
            # print("best_submatches", best_submatches)


            ## Next level
            if lvl < len(sweeped_levels) - 1:
                
                for good_submatch_i in sorted_matches_indexes:
                    print("good_submatch", filter1_matches[good_submatch_i])
                    parent1_data = self.change_pos_dt(G1, filter1_matches[good_submatch_i][:,0], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
                    parent2_data = self.change_pos_dt(G2, filter1_matches[good_submatch_i][:,1], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
                    filter1_subscores = []
                    filter1_submatches = []
                    filter1_submatches_n_assoc = []
                    for i, lowerlevel_match in enumerate(filter1_matches[good_submatch_i]):
                        print("Checking lowerlevel matches for {}\n".format(lowerlevel_match))
                        G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match[0])
                        G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match[1])
                        parents_data = {"match" : lowerlevel_match, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : nodes_id[good_submatch_i]}
                        filter1_submatch, filter1_subscore = match_iteration(G1_neighborhood, G2_neighborhood, lvl + 1, parents_data)
                        filter1_subscores.append(filter1_subscore)
                        filter1_submatches.append(filter1_submatch)
                        filter1_submatches_n_assoc.append(len(filter1_submatch))
                        

            return(filter1_matches, filter1_scores)


        match_iteration(self.graphs[G1_name], self.graphs[G2_name], lvl)
        self.graphs["match"].draw("match", options = None, show = True)


    def only_walls_match_custom(self, G1_name, G2_name):
        full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        self.logger.info("Graph Manager only_walls_match_custom: full_graph_matches - {}".format(len(full_graph_matches)))
        G1_walls = self.graphs[G1_name].filter_graph_by_node_types("Plane")
        G2_walls = self.graphs[G2_name].filter_graph_by_node_types("Plane")
        # self.logger.info("Graph Manager only_walls_match_custom: G2_walls - {}".format(G2_walls.graph.nodes()))
        matches = G1_walls.matchByNodeType(G2_walls)
        self.logger.info("Graph Manager only_walls_match_custom: matches - {}".format(len(matches)))
        matches = self.filter_local_match_with_global(matches, full_graph_matches)
        self.logger.info("Graph Manager only_walls_match_custom: matches in full_graph_matches - {}".format(len(matches)))
        scores = []
        good_matches = []
        for A_categorical in matches:
            data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(self.graphs[G1_name], self.graphs[G2_name], A_categorical, "Geometric_info")
            clipper = Clipper("points&normal")
            clipper.score_pairwise_consistency(data1, data2, A_numerical)
            clipper_match_numerical, score = clipper.solve_clipper()
            clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)
            if score >= SCORE_THR:
                scores.append(score)
                good_matches.append(clipper_match_categorical)
        self.logger.info("Graph Manager only_walls_match_custom: good_matches - {}".format(len(good_matches)))
        matches_list = []
        sorted_matches_indexes = np.argsort(scores)[::-1]
        scores_sorted = []
        success = False
        for i in sorted_matches_indexes:
            success = True
            scores_sorted.append(scores[i])
            match_list = []
            for edge in good_matches[i]:
                edge_dict = {"origin_node" : int(edge[0]), "target_node" : int(edge[1]), "score" : scores[i]}
                match_list.append(edge_dict)
            matches_list.append(match_list)

        self.logger.info("Graph Manager only_walls_match_custom: success - {}".format(success))
        self.logger.info("Graph Manager only_walls_match_custom: matches_list - {}".format(len(matches_list)))
        self.logger.info("Graph Manager only_walls_match_custom: matches_list 2 - {}".format(matches_list))
        self.logger.info("Graph Manager only_walls_match_custom: scores_sorted - {}".format(scores_sorted))

        self.subplots_match(G1_name, G2_name, matches_list[0])

        return(success, matches_list, scores_sorted)


    def filter_local_match_with_global(self, local_matches, global_matches):
        filtered = [local_match for local_match in local_matches if any(all(elem in global_match for elem in local_match) for global_match in global_matches)]
        filtered_as_arrays = [np.array(x, dtype = np.str) for x in filtered]
        return filtered_as_arrays


    def check_match_not_in_list(self, new_match, other_matches):
        if not other_matches:
            return True
        else:
            return( not any(set([tuple(pair) for pair in new_match]) == set([tuple(pair) for pair in match]) for match in other_matches))


    def generate_clipper_input(self, G1, G2, A_categorical, feature_name):
        nodes1, nodes2 = list(set(A_categorical[:,0])), list(set(A_categorical[:,1]))
        data1 = G1.stack_nodes_feature(nodes1, feature_name)
        data2 = G2.stack_nodes_feature(nodes2, feature_name)
        print("nodes1", nodes1)
        print("nodes2", nodes2)
        print("data1", data1)
        print("data2", data2)
        A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
        print("A_numerical", A_numerical)
        return(data1, data2, A_numerical, nodes1, nodes2)


    def change_pos_dt(self, graph, match, in_dt, out_dt):
        original = graph.stack_nodes_feature(match, "Geometric_info")

        if in_dt == out_dt:
            processed = original
        elif in_dt == "points" and out_dt == "points&normal":
            diffs = original - np.concatenate((original[1:], [original[0]]),axis=0)
            normal = diffs / np.sqrt((diffs**2).sum())
            processed = np.concatenate((original, normal),axis=1)

        return(processed)


    def add_parents_data(self, data1, data2, A_numerical, parents_data):
        A_numerical = np.concatenate((A_numerical, [[data1.shape[0], data2.shape[0]]]), axis= 0)
        data1 = np.concatenate((data1, [parents_data["parent1"]]), axis= 0)
        data2 = np.concatenate((data2, [parents_data["parent2"]]), axis= 0)
        return(data1, data2, A_numerical)

    
    def subplots_match(self, g1_name, g2_name, match):

        # fig, axs = plt.subplots(2,2)

        ### Plot base graph
        # plt.axes(axs[0])
        options_base = {'node_color': self.graphs[g1_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.graphs[g1_name].draw(g1_name, options_base, True)

        ### Plot target graph
        # plt.axes(axs[1])
        options_target = {'node_color': self.graphs[g2_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.graphs[g2_name].draw(g2_name, options_target, True)

        ### Plot base graph with match
        # plt.axes(axs[2])
        nodes_base = [pair["origin_node"] for pair in match]
        options_base_matched = self.graphs[g1_name].define_draw_color_from_node_list(options_base, nodes_base, unmatched_color = None, matched_color = "black")
        self.graphs[g1_name].draw("{}_match".format(g1_name), options_base_matched, True)

        ### Plot target graph with match
        # plt.axes(axs[3])
        nodes_target = [pair["target_node"] for pair in match]
        options_target_matched = self.graphs[g2_name].define_draw_color_from_node_list(options_target, nodes_target, unmatched_color = None, matched_color = "black")
        self.graphs[g2_name].draw("{}_match".format(g2_name), options_target_matched, True)