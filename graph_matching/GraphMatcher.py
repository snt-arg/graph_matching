import numpy as np
import itertools
import time
import copy
import json
import os
import pathlib, sys
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .Clipper import Clipper
from .utils import transform_plane_definition, multilist_combinations

graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
sys.path.append(graph_wrapper_dir)
from graph_wrapper.GraphWrapper import GraphWrapper

class GraphMatcher():
    def __init__(self, logger):
        self.graphs = {}
        self.logger = logger

    def set_parameters(self, params):
        self.params = params

    def set_graph_from_dict(self, graph_def, graph_name):
        self.graphs[graph_name] = GraphWrapper(graph_def = graph_def)
        self.logger.info(f"flag {graph_name}, {self.graphs[graph_name].get_attributes_of_all_nodes()}")

    def set_graph_from_wrapper(self, graph_wrapper, graph_name):
        self.graphs[graph_name] = graph_wrapper


    def match(self, G1_name, G2_name):
        self.logger.info("beginning match")

        start_time = time.time()
        swept_levels = self.params["levels"]["name"]
        lvl = 0
        match_graph = GraphWrapper(graph_def={'name': "match",'nodes' : [], 'edges' : []})
        G1_full = self.graphs[G1_name]
        G2_full = self.graphs[G2_name]

        def match_iteration(working_node_ID, lvl):
            ### INTERLEVEL CANDIDATES GENERATION
            if working_node_ID:
                ### Extract every children of the parent higher-level match which belongs to current level
                working_node_attrs = match_graph.get_attributes_of_node(working_node_ID)
                G1_lvl = G1_full.get_neighbourhood_graph(working_node_attrs["match"][0]).filter_graph_by_node_types(swept_levels[lvl])
                G2_lvl = G2_full.get_neighbourhood_graph(working_node_attrs["match"][1]).filter_graph_by_node_types(swept_levels[lvl])
            else:
                ### Extract every node in the hole graph which belongs to the current level
                G1_lvl = G1_full.filter_graph_by_node_types(swept_levels[lvl])
                G2_lvl = G2_full.filter_graph_by_node_types(swept_levels[lvl])
                # options = {'node_color': G1_lvl.define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
                # G1_lvl.draw("S-Graph", options, True)
                # plt.show()
            
            ### Compute all possible node combinations between both subraphs
            all_pairs_categorical = set(itertools.product(G1_lvl.graph.nodes(), G2_lvl.get_nodes_ids()))     
            # all_pairs_categorical = self.filter_local_match_with_global(all_pairs_categorical, full_graph_matches) # TODO include
            if working_node_ID:
                ### Assess GC of each candidate pair with higher-level parent
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, all_pairs_categorical, "Geometric_info")
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["clipper_invariants"][swept_levels[lvl]], self.params, self.logger)
                data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_numerical, working_node_attrs["data_node1"],working_node_attrs["data_node2"])
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[lvl], working_node_attrs["data_node1"]))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[lvl], working_node_attrs["data_node2"]))
                clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                interlevel_scores = M_aux[:,-1][:-1]
                good_pairs = interlevel_scores >= self.params["thresholds"]["local_interlevel"]
                bad_pairs = [not elem for elem in good_pairs]
                filtered_bad_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[bad_pairs], nodes1, nodes2))
                filtered_good_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[good_pairs], nodes1, nodes2))
                interlevel_scores_dict = {list(filtered_good_pairs_categorical)[i]: interlevel_scores[good_pairs][i] for i in range(len(filtered_good_pairs_categorical))}
                
            else:
                interlevel_scores_dict = {list(all_pairs_categorical)[i]: 1. for i in range(len(all_pairs_categorical))}
                filtered_bad_pairs_categorical = []

            ### INTRALEVEL CANDIDATES COMBINATION
            # complete_matches_combinations = G1_lvl.matchByNodeType(G2_lvl)
            interlevel_consistent_combinations = self.delete_list_if_element_inside(G1_lvl.matchByNodeType(G2_lvl), filtered_bad_pairs_categorical)
            filter1_scores = []
            filter1_matches = []
            filter1_lengths = []
            
            for A_categorical in interlevel_consistent_combinations:
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")
                if working_node_ID:
                    data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[lvl], working_node_attrs["data_node1"]))
                    data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[lvl], working_node_attrs["data_node2"]))
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["clipper_invariants"][swept_levels[lvl]], self.params, self.logger)
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                clipper_match_numerical, score = clipper.solve_clipper()
                clipper_match_categorical = set(clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2))

                if score > self.params["thresholds"]["local_intralevel"] and clipper_match_categorical not in filter1_matches:
                    filter1_scores.append(score)
                    filter1_matches.append(clipper_match_categorical)
                    filter1_lengths.append(len(clipper_match_categorical))
            if filter1_scores:
                sorted_matches_indexes = [index for index, val in enumerate(filter1_lengths) if val == max(filter1_lengths)]
                # sorted_matches_indexes = range(len(filter1_lengths))
                for good_submatch_i in sorted_matches_indexes:
                    group_node_id = match_graph.get_total_number_nodes() + 1
                    node_attr = [(group_node_id, {"type": swept_levels[lvl], "match": filter1_matches[good_submatch_i], "merge_lvl" :0,\
                                    "combination_type" : "group", "score_intralevel" : filter1_scores[good_submatch_i]})]
                    if working_node_ID:
                        edges_attr = [(working_node_ID, group_node_id, {})]
                    else:
                        edges_attr = []
                    match_graph.add_subgraph(node_attr, edges_attr)


                    ## Next level
                    if lvl < len(swept_levels) - 1:
                        parent1_data = self.change_pos_dt(G1_full, np.array(list(filter1_matches[good_submatch_i]))[:,0], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl+1]])
                        parent2_data = self.change_pos_dt(G2_full, np.array(list(filter1_matches[good_submatch_i]))[:,1], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl+1]])
                        for i, lowerlevel_match_pair in enumerate(filter1_matches[good_submatch_i]):               
                            existing_nodes = match_graph.find_nodes_by_attrs({"type": swept_levels[lvl], \
                                                "match" : lowerlevel_match_pair, "combination_type" : "pair"})
                            if not existing_nodes:

                                # G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match_pair[0])
                                # G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match_pair[1])
                                
                                pair_node_id = match_graph.get_total_number_nodes() + 1
                                node_attr = [(pair_node_id, {"type": swept_levels[lvl], "match": lowerlevel_match_pair,\
                                                            "combination_type" : "pair", "score_interlevel" : interlevel_scores_dict[lowerlevel_match_pair],\
                                                            "data_node1" : parent1_data[i], "data_node2" : parent2_data[i], "merge_lvl" :0})]
                                
                                edges_attr = [(pair_node_id, group_node_id, {})]

                                match_graph.add_subgraph(node_attr, edges_attr)

                                # next_parents_data = {"match" : lowerlevel_match_pair, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : pair_node_id}
                                # match_iteration(pair_node_id, lvl + 1, next_parents_data)
                                match_iteration(pair_node_id, lvl + 1)

                            else:
                                edges_attr = [(existing_nodes[0], group_node_id, {})]
                                match_graph.add_edges(edges_attr)

            if lvl < len(swept_levels) - 1:
                self.prune_interlevel(match_graph, G1_full, G2_full, swept_levels[lvl:lvl+2])
                self.add_upranted_nodes_by_level(match_graph, G1_full, G2_full, swept_levels[lvl:lvl+2])

            return

        match_iteration(None, lvl)

        node_color = match_graph.define_draw_color_option_by_node_type()
        node_size = match_graph.define_node_size_option_by_combination_type_attr()
        linewidths = match_graph.define_node_linewidth_option_by_combination_type_attr()
        options = {'node_color': node_color, 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    "node_size" : node_size, "linewidths" : linewidths, "edgecolors" : "black"}
        match_graph.draw("match graph", options = options, show = True)

        final_combinations = self.gather_final_combinations_from_match_graph(G1_full, G2_full, match_graph, swept_levels)

        if final_combinations:
            self.logger.info("Found {} good matches!!!".format(len(final_combinations)))
            success = True

            if len(final_combinations) == 1:
                self.logger.info("Only one match succeded with score - {}".format(final_combinations))
            elif len(final_combinations) > 1:
                self.logger.info("{} symmetries detected. Scores - {}".format(len(final_combinations), [match[0] for match in final_combinations]))
            self.subplots_match(G1_name, G2_name, final_combinations[0])

        else:
            success = False
            final_combinations = []
        self.logger.info("Elapsed time in match {}".format(time.time() - start_time))

        return(success, final_combinations)


    def filter_local_match_with_global(self, local_match, global_matches):
        filtered = set([ local_elem for local_elem in local_match if any(local_elem in global_match for global_match in global_matches)])
        return filtered


    def check_match_not_in_list(self, new_match, other_matches):
        if not other_matches:
            return True
        else:
            return( not any(set([tuple(pair) for pair in new_match]) == set([tuple(pair) for pair in match]) for match in other_matches))


    def generate_clipper_input(self, G1, G2, A_categorical, feature_name):
        nodes1, nodes2 = list(np.array(list(A_categorical))[:,0]), list(np.array(list(A_categorical))[:,1])
        data1 = G1.stack_nodes_feature(nodes1, feature_name)
        data2 = G2.stack_nodes_feature(nodes2, feature_name)
        A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
        return(data1, data2, A_numerical, nodes1, nodes2)


    def change_pos_dt(self, graph, node_list, in_dt, out_dt):
        original = graph.stack_nodes_feature(node_list, "Geometric_info")

        if in_dt == out_dt:
            processed = original
        elif in_dt == "points" and out_dt == "points&normal":
            normal = np.repeat(np.array([[0.,0.,1.]]), len(original), axis= 0)
            processed = np.concatenate((original, normal),axis=1)

        return(processed)


    def add_parents_data(self, data1, data2, A_numerical, data_parent1, data_parent2):
        A_numerical_with_parent = np.concatenate((A_numerical, [[data1.shape[0], data2.shape[0]]]), axis= 0, dtype = np.int32)
        data1 = np.concatenate((data1, [data_parent1]), axis= 0, dtype = np.float64)
        data2 = np.concatenate((data2, [data_parent2]), axis= 0, dtype = np.float64)
        return(data1, data2, A_numerical_with_parent)
    

    def add_floor_data(self, data1, data2, A_numerical): ### TODO: Not working. It does not desambiguate
        floor_data = np.repeat([[0,0,0,0,0,1]], len(A_numerical), axis=0)
        A_numerical_with_parent = np.concatenate((A_numerical, [[data1.shape[0], data2.shape[0]]]), axis= 0, dtype = np.int32)
        data1 = np.concatenate(([ data1, [0,0,0,0,0,1]]), axis= 0, dtype = np.float64)
        data2 = np.concatenate(([ data2, [0,0,0,0,0,1]]), axis= 0, dtype = np.float64)
        return(data1, data2, A_numerical_with_parent)
    
    # def delete_floor_data(self, data1, data2, A_numerical):
    #     A_numerical = A_numerical[1:]
    #     self.logger.info("flag data1 {}".format(data1))
    #     data1 = data1[1:]
    #     self.logger.info("flag data1 {}".format(data1))


    def geometric_info_transformation(self, data, level, parent_data):
        if level == "Plane":
            rotation = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
            transformed = transform_plane_definition(data, -parent_data[:3], rotation, self.logger)
        else:
            transformed = data

        return transformed

    
    def subplots_match(self, g1_name, g2_name, match):
        fig, axs = plt.subplots(nrows=2, ncols=2, num="match")
        plt.clf()
        fig.suptitle('Match between {} and {} graphs'.format(g1_name, g2_name))

        ### Plot base graph
        plt.axes(axs[0,0])
        axs[0,0].clear()
        axs[0, 0].set_title('Base graph - {}'.format(g1_name))
        options_base = {'node_color': self.graphs[g1_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.graphs[g1_name].draw(None, options_base, True)

        ### Plot target graph
        plt.axes(axs[0, 1])
        axs[0, 1].clear()
        axs[0, 1].set_title('Target graph - {}'.format(g2_name))
        options_target = {'node_color': self.graphs[g2_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        self.graphs[g2_name].draw(None, options_target, True)

        ### Plot base graph with match
        plt.axes(axs[1,0])
        axs[1,0].clear()
        axs[1,0].set_title('Base graph match - {}'.format(g1_name))
        nodes_base = [pair["origin_node"] for pair in match]
        options_base_matched = self.graphs[g1_name].define_draw_color_from_node_list(options_base, nodes_base, unmatched_color = None, matched_color = "black")
        self.graphs[g1_name].draw(None, options_base_matched, True)

        ### Plot target graph with match
        plt.axes(axs[1,1])
        axs[1,1].clear()
        axs[1,1].set_title('Target graph match - {}'.format(g2_name))
        nodes_target = [pair["target_node"] for pair in match]
        options_target_matched = self.graphs[g2_name].define_draw_color_from_node_list(options_target, nodes_target, unmatched_color = None, matched_color = "black")
        self.graphs[g2_name].draw(None, options_target_matched, True)


    def filter_matches_by_node_type(self, g1, matches, node_type):
        new_matches = []
        for match in matches:
            new_match = []
            for pair in match:
                if g1.get_attributes_of_node(pair[0])["type"] == node_type:
                    new_match.append(pair)
            if new_match not in np.array(new_match):
                new_matches.append(np.array(new_match))
        return new_matches


    def delete_list_if_element_inside(self, lists, filter_elements_list):
        return [list1 for list1 in lists if not any([element in list1 for element in filter_elements_list])]

    def build_matches_msg_from_match_graph(self, match_graph, swept_levels):
        self.logger.info("beginning build_matches_msg_from_match_graph")

        def build_matches_msg_from_match_graph_iteration(local_graph, lvl):
            group_nodes = local_graph.find_nodes_by_attrs({"type": swept_levels[lvl], "combination_type" : "group"})
            group_lvl_upgoing_matches_tuples = []
            for group_node in group_nodes:
                group_node_match = local_graph.get_attributes_of_node(group_node)["match"]
                group_node_score = local_graph.get_attributes_of_node(group_node)["score_intralevel"]
                edges_list_triplet = []
                for edge in group_node_match:
                    edge_triplet = (edge, group_node_score)
                    edges_list_triplet.append(edge_triplet)

                if len(swept_levels) > lvl + 1:
                    group_node_neighbourhood_graph = match_graph.get_neighbourhood_graph(group_node)
                    single_nodes = group_node_neighbourhood_graph.find_nodes_by_attrs({"type": swept_levels[lvl], "combination_type" : "pair"})
                    single_lvl_upgoing_matches_tuples = [[]]
                    for single_node in single_nodes:
                        single_node_neighbourhood_graph = match_graph.get_neighbourhood_graph(single_node)
                        single_node_matches_tuples = build_matches_msg_from_match_graph_iteration(single_node_neighbourhood_graph, lvl+1)

                        prior_single_lvl_upgoing_matches_tuples = copy.deepcopy(single_lvl_upgoing_matches_tuples)
                        single_lvl_upgoing_matches_tuples = []
                        for single_node_match_tuples in single_node_matches_tuples:
                            
                            for single_lvl_upgoing_match_tuples in prior_single_lvl_upgoing_matches_tuples:
                                single_lvl_upgoing_matches_tuples.append(list(set(single_lvl_upgoing_match_tuples).union(set(single_node_match_tuples)).union(edges_list_triplet)))                 

                else:
                    single_lvl_upgoing_matches_tuples = [edges_list_triplet]

                if single_lvl_upgoing_matches_tuples and single_lvl_upgoing_matches_tuples[0]:
                    group_lvl_upgoing_matches_tuples += single_lvl_upgoing_matches_tuples

            return group_lvl_upgoing_matches_tuples

        final_matches_tuples = build_matches_msg_from_match_graph_iteration(match_graph, 0)
        return final_matches_tuples
    

    def gather_final_combinations_from_match_graph(self,G1_full, G2_full, match_graph, swept_levels):
        G1_nodes = G1_full.get_attributes_of_all_nodes()
        G2_nodes = G2_full.get_attributes_of_all_nodes()
        pruned_match_graph = match_graph.filter_graph_by_node_attributes({"merge_lvl":len(swept_levels)-1})
        node_color = pruned_match_graph.define_draw_color_option_by_node_type()
        node_size = pruned_match_graph.define_node_size_option_by_combination_type_attr()
        linewidths = pruned_match_graph.define_node_linewidth_option_by_combination_type_attr()
        options = {'node_color': node_color, 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    "node_size" : node_size, "linewidths" : linewidths, "edgecolors" : "black"}
        pruned_match_graph.draw("pruned match graph", options = options, show = True)

        def gather_final_combinations_from_match_graph_iteration(working_node_ID, lvl):
            if working_node_ID:
                working_node_matches = pruned_match_graph.get_attributes_of_node(working_node_ID)["match"]
                working_node_score = pruned_match_graph.get_attributes_of_node(working_node_ID)["score_intralevel"]
                
                working_node_tuples = [{"origin_node" : int(working_node_match[0]), "target_node" : int(working_node_match[1]), "score" : working_node_score,\
                                "origin_node_attrs" : G1_nodes[working_node_match[0]], "target_node_attrs" : G2_nodes[working_node_match[1]]} for working_node_match in working_node_matches]
                if len(swept_levels) > lvl:
                    neighbour_nodes_IDs = pruned_match_graph.get_neighbourhood_graph(working_node_ID).find_nodes_by_attrs({"type": swept_levels[lvl]})
            else:
                working_node_tuples = []
                if len(swept_levels) > lvl:
                    neighbour_nodes_IDs = pruned_match_graph.find_nodes_by_attrs({"type": swept_levels[lvl]})

            stacked_tuples = []
            if len(swept_levels) > lvl:
                for neighbour_node_ID in neighbour_nodes_IDs:
                    lower_level_nodes_tuples = gather_final_combinations_from_match_graph_iteration(neighbour_node_ID, lvl+1)
                    for lower_level_nodes_tuple in lower_level_nodes_tuples:
                        stacked_tuples.append(working_node_tuples + lower_level_nodes_tuple)

            else:
                stacked_tuples = [working_node_tuples]

            return stacked_tuples
            
        return gather_final_combinations_from_match_graph_iteration(None, 0)



    # def full_graph_affinity_filter(self, G1, G2, swept_levels, matches_tuples_list):
    #     self.logger.info("beginning full_graph_affinity_filter")
    #     test_level = "Plane"
    #     G1_nodes = G1.get_attributes_of_all_nodes()
    #     G2_nodes = G2.get_attributes_of_all_nodes()
    #     clipper_matches_msg = []
    #     quality_success_percentage_list = []
    #     for match_tuples in matches_tuples_list:
    #         basis_level_match = [pair[0] for pair in match_tuples if G1_nodes[pair[0][0]]["type"] == test_level]
    #         top_level_match = [pair[0] for pair in match_tuples if G1_nodes[pair[0][0]]["type"] == swept_levels[0]][0]
    #         top_level_info = [np.array(G1_nodes[top_level_match[0]]["Geometric_info"]), np.array(G2_nodes[top_level_match[1]]["Geometric_info"])]
    #         if basis_level_match:
    #             data1, data2, basis_level_match_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, basis_level_match, "Geometric_info")
    #             data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[-1], top_level_info[0]))
    #             data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[-1], top_level_info[1]))
    #             clipper = Clipper(self.params["levels"]["datatype"][test_level], 1, self.params, self.logger)
    #             clipper.score_pairwise_consistency(data1, data2, basis_level_match_numerical)
    #             consistency_avg = clipper.get_score_all_inital_u()

    #             if consistency_avg > self.params["thresholds"]["global"]:
    #                 edges_dict_list = []
    #                 for edge in match_tuples:
    #                     edge_dict = {"origin_node" : int(edge[0][0]), "target_node" : int(edge[0][1]), "score" : edge[1],\
    #                             "origin_node_attrs" : G1_nodes[edge[0][0]], "target_node_attrs" : G2_nodes[edge[0][1]]}
    #                     edges_dict_list.append(edge_dict)
    #                 clipper_matches_msg += [(consistency_avg, edges_dict_list)]
    #                 quality_success_percentage_list.append(consistency_avg)

    #         else:
    #             self.logger.info("There was no node of test_level in the graph")
    #     self.logger.info("Affinity check: {} out of {} candidates passed the final check"\
    #                     .format(len(clipper_matches_msg), len(matches_tuples_list)))
            
    #     clipper_matches_msg_sorted = [clipper_matches_msg[i] for i in np.argsort([match[0] for match in clipper_matches_msg])[::-1]]
    #     return clipper_matches_msg_sorted

    def symmetry_detection(self, candidates):
        X = np.array([match["consistency_avg"] for match in candidates])
        # self.logger.info("X {}".format(X))
        # X_fit = StandardScaler().fit_transform(X.reshape(0, 1))
        X_fit = np.expand_dims(X, axis=1)
        db = DBSCAN(eps=self.params["dbscan"]["eps"], min_samples=self.params["dbscan"]["min_samples"]).fit(X_fit)
        labels = db.labels_
        # self.logger.info("flag labels {} best {}".format(labels, labels[np.argmax(X)]))
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        best_cluster_candidates = [candidates[i] for i,label in enumerate(labels) if label==labels[np.argmax(X)]]

        return best_cluster_candidates
    

    def prune_interlevel(self,match_graph, G1_full, G2_full, merged_levels):

        higher_level_group_nodes = list(match_graph.filter_graph_by_node_types(merged_levels[0])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids())
        # self.logger.info("higher_level_group_nodes {}".format(higher_level_group_nodes))
        consistent_combinations = []
        for node in higher_level_group_nodes:
            new_consistent_combinations = self.merge_lower_level_groups(match_graph, G1_full, G2_full, node, merged_levels)
            if new_consistent_combinations:
                consistent_combinations += new_consistent_combinations
            
        self.select_high_level_groups(match_graph, consistent_combinations, merged_levels)        

    
    def merge_lower_level_groups(self, match_graph, G1_full, G2_full, working_node_ID, merged_levels):
        higher_level_single_pairs_nodes = list(match_graph.get_neighbourhood_graph(working_node_ID)\
                                                            .filter_graph_by_node_types(merged_levels[0])\
                                                            .filter_graph_by_node_attributes({"combination_type" : "pair"})\
                                                            .get_nodes_ids())
        lower_level_group_nodes = [list(match_graph.get_neighbourhood_graph(node)\
                                                    .filter_graph_by_node_types(merged_levels[1])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids()) for node in higher_level_single_pairs_nodes]
        
        combinations = multilist_combinations(lower_level_group_nodes)
        parent_node_attrs = match_graph.get_attributes_of_node(higher_level_single_pairs_nodes[0])
        # parent1_data = self.change_pos_dt(G1_full, [parent_node_attrs["match"][0]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        # parent2_data = self.change_pos_dt(G2_full, [parent_node_attrs["match"][1]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        consistent_combinations = []
        for combination in combinations:
            A_categorical = set()
            for node in combination:
                A_categorical.update(match_graph.get_attributes_of_node(node)["match"])
            data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")
            data1 = self.geometric_info_transformation(data1, merged_levels[1], G1_full.get_attributes_of_node(parent_node_attrs["match"][0])["Geometric_info"])
            data2 = self.geometric_info_transformation(data2, merged_levels[1], G2_full.get_attributes_of_node(parent_node_attrs["match"][1])["Geometric_info"])
            clipper = Clipper(self.params["levels"]["datatype"][merged_levels[1]], self.params["levels"]["clipper_invariants"][merged_levels[1]], self.params, self.logger)
            clipper.score_pairwise_consistency(data1, data2, A_numerical)
            consistency_avg = clipper.get_score_all_inital_u()
            if consistency_avg >= self.params["thresholds"]["global"]:
                consistent_combinations.append({"consistency_avg":consistency_avg,"lower_level_nodes_IDs": combination,"match":A_categorical, "higher_level_node_ID":working_node_ID})

        return consistent_combinations


    def select_high_level_groups(self, match_graph, consistent_combinations, merged_levels):
        if consistent_combinations:
            best_combinations = self.symmetry_detection(consistent_combinations)

            for best_combination in best_combinations:
                best_combination_node_id = match_graph.get_total_number_nodes() + 1
                node_attr = [(best_combination_node_id, {"type": merged_levels[1], "match": best_combination["match"], "merge_lvl" :1,\
                                            "combination_type" : "group", "score_intralevel" : best_combination["consistency_avg"]})]
                edges_attr = [(lower_level_node, best_combination_node_id, {}) for lower_level_node in best_combination["lower_level_nodes_IDs"]]
                edges_attr.append((best_combination["higher_level_node_ID"], best_combination_node_id, {}))
                match_graph.add_subgraph(node_attr, edges_attr)

                match_graph.set_node_attributes("merge_lvl", {best_combination["higher_level_node_ID"]:1})

    
    def add_upranted_nodes_by_level(self, match_graph, G1_full, G2_full, swept_levels):
        G1_level_pair_nodes_all = G1_full.find_nodes_by_attrs({"type": swept_levels[1]})
        G2_level_pair_nodes_all = G2_full.find_nodes_by_attrs({"type": swept_levels[1]})
        G2_level_pair_nodes_all_unparented = [node for node in G2_level_pair_nodes_all if not G2_full.get_neighbourhood_graph(node).filter_graph_by_node_attributes({"type": swept_levels[0]})]
        if G2_level_pair_nodes_all_unparented:
            merged_nodes = match_graph.find_nodes_by_attrs({"type": swept_levels[1], "merge_lvl": 1})
            combinations = []

            for merged_node in merged_nodes:
                merged_node_match = match_graph.get_attributes_of_node(merged_node)["match"]
                G1_matched_nodes = np.array(list(merged_node_match))[:,0]
                G2_matched_nodes = np.array(list(merged_node_match))[:,1]
                G1_wild_nodes = [x for x in G1_level_pair_nodes_all if x not in G1_matched_nodes]
                # self.logger.info("flag G1_wild_nodes {}".format(G1_wild_nodes))
                G2_wild_nodes = [x for x in G2_level_pair_nodes_all_unparented if x not in G2_matched_nodes]
                # self.logger.info("flag G2_wild_nodes {}".format(G2_wild_nodes))
                wild_nodes_combination = multilist_combinations([G1_wild_nodes, G2_wild_nodes])
                # self.logger.info("flag wild_nodes_combination {}".format(wild_nodes_combination))

                ### Use clipper utility function to compute consistency
                parent1_data = G1_full.get_attributes_of_node(G1_matched_nodes[0])["Geometric_info"]
                parent2_data = G2_full.get_attributes_of_node(G2_matched_nodes[0])["Geometric_info"]
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, wild_nodes_combination, "Geometric_info")
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[1]], self.params["levels"]["clipper_invariants"][swept_levels[1]], self.params, self.logger)
                data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_numerical, parent1_data, parent2_data)
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[1], parent1_data))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[1], parent2_data))
                clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                interlevel_scores = M_aux[:,-1][:-1]
                good_pairs = interlevel_scores >= self.params["thresholds"]["local_interlevel"]
                bad_pairs = [not elem for elem in good_pairs]
                filtered_bad_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[bad_pairs], nodes1, nodes2))
                filtered_good_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[good_pairs], nodes1, nodes2))
                # interlevel_scores_dict = {list(filtered_good_pairs_categorical)[i]: interlevel_scores[good_pairs][i] for i in range(len(filtered_good_pairs_categorical))}
                
                merged_node_match.update(filtered_good_pairs_categorical)

                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, merged_node_match, "Geometric_info")
                data1 = self.geometric_info_transformation(data1, swept_levels[1], parent1_data)
                data2 = self.geometric_info_transformation(data2, swept_levels[1], parent2_data)
                data1, data2, A_numerical = self.add_floor_data(data1, data2, A_numerical)
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[1]], self.params["levels"]["clipper_invariants"][swept_levels[1]], self.params, self.logger)
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                consistency_avg = clipper.get_score_all_inital_u()
                combinations.append({"consistency_avg":consistency_avg, "match": merged_node_match, "base_node_ID": merged_node})

            best_combinations = self.symmetry_detection(combinations) ### TODO: Detect and fix when this line crashes

            for combination in combinations:
                if combination in best_combinations:
                    match_graph.set_node_attributes("consistency_avg", {combination["base_node_ID"]: combination["consistency_avg"]})
                    match_graph.set_node_attributes("match", {combination["base_node_ID"]: combination["match"]})
                    # self.logger.info("flag good combination {}".format(combination))
                else:
                    match_graph.remove_nodes([combination["base_node_ID"]])
            