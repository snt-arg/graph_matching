import numpy as np
import itertools
import time
import copy
import json
import os
import pathlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .GraphWrapper import GraphWrapper
from .Clipper import Clipper
from .utils import transform_plane_definition, multilist_combinations


class GraphMatcher():
    def __init__(self, logger):
        self.graphs = {}
        self.logger = logger

    def set_parameters(self, params):
        self.params = params


    def setGraph(self, graph_def):
        self.graphs[graph_def['name']] = GraphWrapper(graph_def = graph_def)


    def match_custom(self, G1_name, G2_name):
        self.logger.info("beginning match_custom")

        start_time = time.time()
        sweeped_levels = self.params["levels"]["name"]
        self.params["levels"]["datatype"] = self.params["levels"]["datatype"]
        self.params["levels"]["clipper_invariants"] = self.params["levels"]["clipper_invariants"]
        # full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        lvl = 0
        match_graph = GraphWrapper(graph_def={'name': "match",'nodes' : [], 'edges' : []})
        G1_full = self.graphs[G1_name]
        G2_full = self.graphs[G2_name]

        ### Detect unparented nodes
        unparented_nodes = {}
        for i, sweeped_level in enumerate(sweeped_levels[1:]):
            unparented_nodes[sweeped_level] = {"G1": [], "G2": []}
            for node in G1_full.filter_graph_by_node_types(sweeped_level).get_nodes_ids():
                if len(G1_full.get_neighbourhood_graph(node).filter_graph_by_node_types(sweeped_levels[i]).get_nodes_ids()) == 0:
                    unparented_nodes[sweeped_level]["G1"].append(node)
            for node in G2_full.filter_graph_by_node_types(sweeped_level).get_nodes_ids():
                if len(G2_full.get_neighbourhood_graph(node).filter_graph_by_node_types(sweeped_levels[i]).get_nodes_ids()) == 0:
                    unparented_nodes[sweeped_level]["G2"].append(node)

        self.logger.info("unparented_nodes {}".format(unparented_nodes))

        def match_iteration(working_node_ID, lvl):
            if working_node_ID:
                working_node_attrs = match_graph.get_attributes_of_node(working_node_ID)
                G1_lvl = G1_full.get_neighbourhood_graph(working_node_attrs["match"][0]).filter_graph_by_node_types(sweeped_levels[lvl])
                G2_lvl = G2_full.get_neighbourhood_graph(working_node_attrs["match"][1]).filter_graph_by_node_types(sweeped_levels[lvl])
            else:
                G1_lvl = G1_full.filter_graph_by_node_types(sweeped_levels[lvl])
                G2_lvl = G2_full.filter_graph_by_node_types(sweeped_levels[lvl])

            all_pairs_categorical = self.get_all_possible_match_pairs(G1_lvl.graph.nodes(), G2_lvl.get_nodes_ids())            
            # all_pairs_categorical = self.filter_local_match_with_global(all_pairs_categorical, full_graph_matches) # TODO Fix
            if working_node_ID:
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, all_pairs_categorical, "Geometric_info")
                clipper = Clipper(self.params["levels"]["datatype"][sweeped_levels[lvl]], self.params["levels"]["clipper_invariants"][sweeped_levels[lvl]], self.params, self.logger)
                data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_numerical, working_node_attrs["data_node1"],working_node_attrs["data_node2"])
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, sweeped_levels[lvl], working_node_attrs["data_node1"]))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, sweeped_levels[lvl], working_node_attrs["data_node2"]))
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

            matches = self.delete_list_if_element_inside(G1_lvl.matchByNodeType(G2_lvl), filtered_bad_pairs_categorical)
            filter1_scores = []
            filter1_matches = []
            filter1_lengths = []
            
            for A_categorical in matches:
                self.logger.info("flag 1 A_categorical {}".format(A_categorical))
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")
                if working_node_ID:
                    self.logger.info("flag 1 parents {}".format(working_node_attrs["match"]))
                    self.logger.info("flag 1 parents data {} {}".format(working_node_attrs["data_node1"], working_node_attrs["data_node2"]))
                    self.logger.info("flag 1 trns level {}".format(sweeped_levels[lvl]))
                    data1 = copy.deepcopy(self.geometric_info_transformation(data1, sweeped_levels[lvl], working_node_attrs["data_node1"]))
                    self.logger.info("flag 1 data1 {}".format(data1))
                    data2 = copy.deepcopy(self.geometric_info_transformation(data2, sweeped_levels[lvl], working_node_attrs["data_node2"]))
                clipper = Clipper(self.params["levels"]["datatype"][sweeped_levels[lvl]], self.params["levels"]["clipper_invariants"][sweeped_levels[lvl]], self.params, self.logger)
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                clipper_match_numerical, score = clipper.solve_clipper()
                self.logger.info("flag 1 score {}".format(score))
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
                    node_attr = [(group_node_id, {"type": sweeped_levels[lvl], "match": filter1_matches[good_submatch_i],\
                                    "combination_type" : "group", "score_intralevel" : filter1_scores[good_submatch_i]})]
                    if working_node_ID:
                        edges_attr = [(working_node_ID, group_node_id)]
                    else:
                        edges_attr = []
                    match_graph.add_subgraph(node_attr, edges_attr)


                    ## Next level
                    if lvl < len(sweeped_levels) - 1:
                        parent1_data = self.change_pos_dt(G1_full, np.array(list(filter1_matches[good_submatch_i]))[:,0], self.params["levels"]["datatype"][sweeped_levels[lvl]], self.params["levels"]["datatype"][sweeped_levels[lvl+1]])
                        parent2_data = self.change_pos_dt(G2_full, np.array(list(filter1_matches[good_submatch_i]))[:,1], self.params["levels"]["datatype"][sweeped_levels[lvl]], self.params["levels"]["datatype"][sweeped_levels[lvl+1]])
                        for i, lowerlevel_match_pair in enumerate(filter1_matches[good_submatch_i]):               
                            existing_nodes = match_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], \
                                                "match" : lowerlevel_match_pair, "combination_type" : "pair"})
                            if not existing_nodes:

                                # G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match_pair[0])
                                # G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match_pair[1])
                                
                                pair_node_id = match_graph.get_total_number_nodes() + 1
                                node_attr = [(pair_node_id, {"type": sweeped_levels[lvl], "match": lowerlevel_match_pair,\
                                                            "combination_type" : "pair", "score_interlevel" : interlevel_scores_dict[lowerlevel_match_pair],\
                                                            "data_node1" : parent1_data[i], "data_node2" : parent2_data[i]})]
                                
                                edges_attr = [(pair_node_id, group_node_id)]

                                match_graph.add_subgraph(node_attr, edges_attr)

                                # next_parents_data = {"match" : lowerlevel_match_pair, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : pair_node_id}
                                # match_iteration(pair_node_id, lvl + 1, next_parents_data)
                                match_iteration(pair_node_id, lvl + 1)

                            else:
                                edges_attr = [(existing_nodes[0], group_node_id)]
                                match_graph.add_subgraph([], edges_attr)

            return

        match_iteration(None, lvl)
        self.prune_interlevel(match_graph, G1_full, G2_full, ["Finite Room", "Plane"])
                
        self.logger.info("Elapsed time in custom match {}".format(time.time() - start_time))
        options = {'node_color': match_graph.define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    "node_size" : match_graph.define_node_size_option_by_combination_type_attr()}
        match_graph.draw("match graph", options = options, show = True)
        matches_tuples_list = self.build_matches_msg_from_match_graph(match_graph, sweeped_levels)
        final_matches_msg = self.full_graph_affinity_filter(self.graphs[G1_name], self.graphs[G2_name], sweeped_levels, matches_tuples_list)

        if final_matches_msg:
            self.logger.info("Found {} good matches!!!".format(len(final_matches_msg)))
            success = True
            match_lengths = [len(match) for match in final_matches_msg]
            biggest_maches_i = [index for index, val in enumerate(match_lengths) if val == max(match_lengths)]
            # best_match_i = np.argmax([final_matches_msg[i][0] for i in biggest_maches_i])
            # self.subplots_match(G1_name, G2_name, final_matches_msg[biggest_maches_i[best_match_i]][1])
            biggest_consistent_matches = [final_matches_msg[i] for i in biggest_maches_i]
            final_matches_msg_selection = self.symmetry_detection(biggest_consistent_matches)

            if len(final_matches_msg_selection) == 1:
                self.logger.info("Only one match succeded with score - {}".format(final_matches_msg_selection[0][0]))
            elif len(final_matches_msg_selection) > 1:
                self.logger.info("{} symmetries detected. Scores - {}".format(len(final_matches_msg_selection), [match[0] for match in final_matches_msg_selection]))
            self.subplots_match(G1_name, G2_name, final_matches_msg_selection[0][1])

        else:
            success = False
            final_matches_msg_selection = []

        return(success, final_matches_msg_selection)


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
        self.logger.info("flag node_list {}".format(node_list))
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
        # self.logger.info("Graph Manager filter_matches_by_node_type 0 - {}".format(matches))
        new_matches = []
        for match in matches:
            new_match = []
            for pair in match:
                if g1.get_attributes_of_node(pair[0])["type"] == node_type:
                    new_match.append(pair)
            if new_match not in np.array(new_match):
                new_matches.append(np.array(new_match))
        # self.logger.info("Graph Manager filter_matches_by_node_type 1 - {}".format(new_matches))
        return new_matches

    def get_all_possible_match_pairs(self, list1, list2):
        return set(itertools.product(list1, list2))

    def delete_list_if_element_inside(self, lists, filter_elements_list):
        return [list1 for list1 in lists if not any([element in list1 for element in filter_elements_list])]

    def build_matches_msg_from_match_graph(self, match_graph, sweeped_levels):
        self.logger.info("beginning build_matches_msg_from_match_graph")

        def build_matches_msg_from_match_graph_iteration(local_graph, lvl):
            group_nodes = local_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], "combination_type" : "group"})
            group_lvl_upgoing_matches_tuples = []
            for group_node in group_nodes:
                group_node_match = local_graph.get_attributes_of_node(group_node)["match"]
                group_node_score = local_graph.get_attributes_of_node(group_node)["score_intralevel"]
                edges_list_triplet = []
                for edge in group_node_match:
                    edge_triplet = (edge, group_node_score)
                    edges_list_triplet.append(edge_triplet)

                if len(sweeped_levels) > lvl + 1:
                    group_node_neighbourhood_graph = match_graph.get_neighbourhood_graph(group_node)
                    single_nodes = group_node_neighbourhood_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], "combination_type" : "pair"})
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


    def full_graph_affinity_filter(self, G1, G2, sweeped_levels, matches_tuples_list):
        self.logger.info("beginning full_graph_affinity_filter")
        test_level = "Plane"
        G1_nodes = G1.get_attributes_of_all_nodes()
        G2_nodes = G2.get_attributes_of_all_nodes()
        clipper_matches_msg = []
        quality_success_percentage_list = []
        for match_tuples in matches_tuples_list:
            basis_level_match = [pair[0] for pair in match_tuples if G1_nodes[pair[0][0]]["type"] == test_level]
            top_level_match = [pair[0] for pair in match_tuples if G1_nodes[pair[0][0]]["type"] == sweeped_levels[0]][0]
            top_level_info = [np.array(G1_nodes[top_level_match[0]]["Geometric_info"]), np.array(G2_nodes[top_level_match[1]]["Geometric_info"])]
            if basis_level_match:
                data1, data2, basis_level_match_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, basis_level_match, "Geometric_info")
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, sweeped_levels[-1], top_level_info[0]))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, sweeped_levels[-1], top_level_info[1]))
                clipper = Clipper(self.params["levels"]["datatype"][test_level], 1, self.params, self.logger)
                clipper.score_pairwise_consistency(data1, data2, basis_level_match_numerical)
                consistency_avg = clipper.get_score_all_inital_u()
                # self.logger.info("data1 {}".format(data1))
                # self.logger.info("data2 {}".format(data2))
                # self.logger.info("data1 - data2 {}".format(data1 - data2))

                if consistency_avg > self.params["thresholds"]["global"]:
                    edges_dict_list = []
                    for edge in match_tuples:
                        edge_dict = {"origin_node" : int(edge[0][0]), "target_node" : int(edge[0][1]), "score" : edge[1],\
                                "origin_node_attrs" : G1_nodes[edge[0][0]], "target_node_attrs" : G2_nodes[edge[0][1]]}
                        edges_dict_list.append(edge_dict)
                    clipper_matches_msg += [(consistency_avg, edges_dict_list)]
                    quality_success_percentage_list.append(consistency_avg)

            else:
                self.logger.info("There was no node of test_level in the graph")
        self.logger.info("Affinity check: {} out of {} candidates passed the final check"\
                        .format(len(clipper_matches_msg), len(matches_tuples_list)))
            
        clipper_matches_msg_sorted = [clipper_matches_msg[i] for i in np.argsort([match[0] for match in clipper_matches_msg])[::-1]]
        return clipper_matches_msg_sorted

    def symmetry_detection(self, candidates):
        X = np.array([match[0] for match in candidates])
        self.logger.info("X {}".format(X))
        # X_fit = StandardScaler().fit_transform(X.reshape(0, 1))
        X_fit = np.expand_dims(X, axis=1)
        db = DBSCAN(eps=self.params["dbscan"]["eps"], min_samples=self.params["dbscan"]["min_samples"]).fit(X_fit)
        labels = db.labels_
        self.logger.info("labels {}".format(labels))
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)

        best_cluster_candidates = [candidates[i] for i,label in enumerate(labels) if label==0]

        return best_cluster_candidates
    

    def prune_interlevel(self,match_graph, G1_full, G2_full, merged_levels):

        higher_level_group_nodes = list(match_graph.filter_graph_by_node_types(merged_levels[0])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids())
        # self.logger.info("higher_level_group_nodes {}".format(higher_level_group_nodes))
        for node in higher_level_group_nodes:
            self.merge_lower_level_groups(match_graph, G1_full, G2_full, node, merged_levels)


    # def select_high_level_groups(match_graph, G1_full, G2_full, working_node_ID, merged_levels)):
        

    
    def merge_lower_level_groups(self, match_graph, G1_full, G2_full, working_node_ID, merged_levels):
        higher_level_single_pairs_nodes = list(match_graph.get_neighbourhood_graph(working_node_ID)\
                                                            .filter_graph_by_node_types(merged_levels[0])\
                                                            .filter_graph_by_node_attributes({"combination_type" : "pair"})\
                                                            .get_nodes_ids())
        # self.logger.info("higher_level_single_pairs_nodes {}".format(higher_level_single_pairs_nodes))
        lower_level_group_nodes = [list(match_graph.get_neighbourhood_graph(node)\
                                                    .filter_graph_by_node_types(merged_levels[1])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids()) for node in higher_level_single_pairs_nodes]
        
        combinations = multilist_combinations(lower_level_group_nodes)
        # self.logger.info("lower_level_group_nodes {}".format(lower_level_group_nodes))
        # self.logger.info("combinations {}".format(combinations))
        parent_node_attrs = match_graph.get_attributes_of_node(higher_level_single_pairs_nodes[0])
        parent1_data = self.change_pos_dt(G1_full, [parent_node_attrs["match"][0]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        parent2_data = self.change_pos_dt(G2_full, [parent_node_attrs["match"][1]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        consistent_score_and_combinations = []
        for i, combination in enumerate(combinations):
            self.logger.info("flag 2 combination {}".format(combination))
            A_categorical = set()
            # [A_categorical.append(match_graph.get_attributes_of_node(node)["match"]) for node in combination]
            # A_categorical = A_categorical[0]
            for node in combination:
                A_categorical.update(match_graph.get_attributes_of_node(node)["match"])
            self.logger.info("flag 2 A_categorical {}".format(A_categorical))
            self.logger.info("flag 2 parents {}".format(parent_node_attrs["match"]))
            self.logger.info("flag 2 parents data {} {}".format(G1_full.get_attributes_of_node(parent_node_attrs["match"][0])["Geometric_info"], G2_full.get_attributes_of_node(parent_node_attrs["match"][1])["Geometric_info"]))
            self.logger.info("flag 2 trns level {}".format(merged_levels[0]))
            data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")
            data1 = self.geometric_info_transformation(data1, merged_levels[1], G1_full.get_attributes_of_node(parent_node_attrs["match"][0])["Geometric_info"])
            self.logger.info("flag 2 data1 {}".format(data1))
            data2 = self.geometric_info_transformation(data2, merged_levels[1], G2_full.get_attributes_of_node(parent_node_attrs["match"][1])["Geometric_info"])
            clipper = Clipper(self.params["levels"]["datatype"][merged_levels[1]], self.params["levels"]["clipper_invariants"][merged_levels[1]], self.params, self.logger)
            clipper.score_pairwise_consistency(data1, data2, A_numerical)
            consistency_avg = clipper.get_score_all_inital_u()
            self.logger.info("flag 2 consistency_avg {}".format(consistency_avg))
            if consistency_avg >= self.params["thresholds"]["global"]:
                consistent_score_and_combinations.append([consistency_avg,combination, i])

        self.logger.info("consistent_combinations {}".format(consistent_score_and_combinations))
        if consistent_score_and_combinations:
            best_combinations = self.symmetry_detection(consistent_score_and_combinations)
            self.logger.info("flag 2 best_combinations {}".format(best_combinations))


