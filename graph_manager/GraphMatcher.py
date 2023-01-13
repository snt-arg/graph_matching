import numpy as np
import itertools
import time
from .GraphManager import GraphManager
from .Clipper import Clipper
import matplotlib.pyplot as plt

INTRALEVEL_CLIPPER_THR = 0.7
INTERLEVEL_CLIPPER_THR = 0.7


class GraphMatcher():
    def __init__(self, logger):
        self.graphs = {}
        self.logger = logger


    def setGraph(self, graph_def):
        self.graphs[graph_def['name']] = GraphManager(graph_def = graph_def)


    def match_custom(self, G1_name, G2_name):
        start_time = time.time()
        sweeped_levels = ["floor", "Finite Room", "Plane"]
        sweeped_levels_dt = {"floor" : "points", "Finite Room" : "points", "Plane": "points&normal"}
        sweeped_levels = ["Finite Room", "Plane"]
        full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        lvl = 0
        match_graph = GraphManager(graph_def={'name': "match",'nodes' : [], 'edges' : []})

        def match_iteration(G1, G2, lvl, parents_data = None):
            G1_lvl = G1.filter_graph_by_node_types(sweeped_levels[lvl])
            G2_lvl = G2.filter_graph_by_node_types(sweeped_levels[lvl])

            all_pairs_categorical = self.get_all_possible_match_pairs(G1_lvl.graph.nodes(), G2_lvl.graph.nodes())
            all_pairs_categorical = self.filter_local_match_with_global(all_pairs_categorical, full_graph_matches)
            if parents_data:
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, all_pairs_categorical, "Geometric_info")
                clipper = Clipper(sweeped_levels_dt[sweeped_levels[lvl]])
                data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_numerical, parents_data)
                clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                interlevel_scores = M_aux[:,-1][:-1]
                good_pairs = interlevel_scores >= INTERLEVEL_CLIPPER_THR
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
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, A_categorical, "Geometric_info")
                clipper = Clipper(sweeped_levels_dt[sweeped_levels[lvl]])
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                clipper_match_numerical, score = clipper.solve_clipper()
                clipper_match_categorical = set(clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2))

                if score > INTRALEVEL_CLIPPER_THR and clipper_match_categorical not in filter1_matches:
                    filter1_scores.append(score)
                    filter1_matches.append(clipper_match_categorical)
                    filter1_lengths.append(len(clipper_match_categorical))

            if filter1_scores:
                sorted_matches_indexes = list(np.squeeze(np.argwhere(filter1_lengths == np.amax(filter1_lengths)), axis= 1))
                nodes_id = {}
                for good_submatch_i in sorted_matches_indexes:
                    group_node_id = match_graph.get_total_number_nodes() + 1
                    node_attr = [(group_node_id, {"type": sweeped_levels[lvl], "match": filter1_matches[good_submatch_i],\
                                    "combination_type" : "group", "score_intralevel" : filter1_scores[good_submatch_i]})]
                    if parents_data:
                        edges_attr = [(parents_data["id"], group_node_id)]
                    else:
                        edges_attr = []
                    match_graph.add_subgraph(node_attr, edges_attr)


                    ## Next level
                    if lvl < len(sweeped_levels) - 1:
                        parent1_data = self.change_pos_dt(G1, np.array(list(filter1_matches[good_submatch_i]))[:,0], sweeped_levels_dt[sweeped_levels[lvl]], sweeped_levels_dt[sweeped_levels[lvl+1]])
                        parent2_data = self.change_pos_dt(G2, np.array(list(filter1_matches[good_submatch_i]))[:,1], sweeped_levels_dt[sweeped_levels[lvl]], sweeped_levels_dt[sweeped_levels[lvl+1]])
                        for i, lowerlevel_match_pair in enumerate(filter1_matches[good_submatch_i]):                             
                            existing_nodes = match_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], \
                                                "match" : lowerlevel_match_pair, "combination_type" : "pair"})
                            if not existing_nodes:
                                G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match_pair[0])
                                G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match_pair[1])
                                
                                pair_node_id = match_graph.get_total_number_nodes() + 1
                                node_attr = [(pair_node_id, {"type": sweeped_levels[lvl], "match": lowerlevel_match_pair,\
                                                "combination_type" : "pair", "score_interlevel" : interlevel_scores_dict[lowerlevel_match_pair]})]
                                
                                edges_attr = [(pair_node_id, group_node_id)]

                                match_graph.add_subgraph(node_attr, edges_attr)

                                next_parents_data = {"match" : lowerlevel_match_pair, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : pair_node_id}
                                match_iteration(G1_neighborhood, G2_neighborhood, lvl + 1, next_parents_data)

                            else:
                                edges_attr = [(existing_nodes[0], group_node_id)]
                                match_graph.add_subgraph([], edges_attr)

            return


        match_iteration(self.graphs[G1_name], self.graphs[G2_name], lvl)
        self.logger.info("Elapsed time in custom match {}".format(time.time() - start_time))
        options = {'node_color': match_graph.define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    "node_size" : match_graph.define_node_size_option_by_combination_type_attr()}
        match_graph.draw("match graph", options = options, show = True)
        matches_msg_list, matches_tuples_list = self.build_matches_msg_from_match_graph(match_graph, sweeped_levels)
        final_matches_msg = self.full_graph_clipper_filter(self.graphs[G1_name], self.graphs[G2_name], sweeped_levels[-1],\
                                                                    sweeped_levels_dt[sweeped_levels[-1]], matches_msg_list, matches_tuples_list)

        if final_matches_msg:
            success = True
            best_match_i = np.argmax([match[0] for match in final_matches_msg])
            self.logger.info("Found {} good matches!!!".format(len(final_matches_msg)))
            self.subplots_match(G1_name, G2_name, final_matches_msg[best_match_i][1])
        else:
            success = False

        return(success, final_matches_msg)


    def only_walls_match_custom(self, G1_name, G2_name):
        full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        self.logger.info("Graph Manager only_walls_match_custom: full_graph_matches - {}".format(len(full_graph_matches)))
        # G1_walls = self.graphs[G1_name].filter_graph_by_node_types("Plane")
        # G2_walls = self.graphs[G2_name].filter_graph_by_node_types("Plane")
        # matches = G1_walls.matchByNodeType(G2_walls)
        # self.logger.info("Graph Manager only_walls_match_custom: matches - {}".format(len(matches)))
        # matches = self.filter_local_match_with_global(matches, full_graph_matches)
        matches = self.filter_matches_by_node_type(self.graphs[G1_name], full_graph_matches, "Plane")
        self.logger.info("Graph Manager only_walls_match_custom: matches in full_graph_matches - {}".format(len(matches)))
        scores = []
        good_matches = []
        for A_categorical in matches:
            data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(self.graphs[G1_name], self.graphs[G2_name], A_categorical, "Geometric_info")
            clipper = Clipper("points&normal")
            clipper.score_pairwise_consistency(data1, data2, A_numerical)
            clipper_match_numerical, score = clipper.solve_clipper()
            clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)
            if score >= INTRALEVEL_CLIPPER_THR:
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

        if success:
            self.subplots_match(G1_name, G2_name, matches_list[0])

        return(success, matches_list, scores_sorted)


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


    def change_pos_dt(self, graph, match, in_dt, out_dt):
        original = graph.stack_nodes_feature(match, "Geometric_info")

        if in_dt == out_dt:
            processed = original
        elif in_dt == "points" and out_dt == "points&normal":
            distance = np.sqrt((original**2).sum())
            if distance:
                normal = original / distance
            else:
                normal = np.array([[0.,0.,0.]])
            processed = np.concatenate((original, normal),axis=1)

        return(processed)


    def add_parents_data(self, data1, data2, A_numerical, parents_data):
        A_numerical_with_parent = np.concatenate((A_numerical, [[data1.shape[0], data2.shape[0]]]), axis= 0, dtype = np.int32)
        data1 = np.concatenate((data1, [parents_data["parent1"]]), axis= 0, dtype = np.float64)
        data2 = np.concatenate((data2, [parents_data["parent2"]]), axis= 0, dtype = np.float64)
        return(data1, data2, A_numerical_with_parent)

    
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
                if g1.graph.nodes(data=True)[pair[0]]["type"] == node_type:
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
        matches_msgs = []

        def build_matches_msg_from_match_graph_iteration(local_graph, lvl, matches_msgs, matches_tuples):
            group_nodes = local_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], "combination_type" : "group"})
            group_lvl_upgoing_matches_msgs = []
            group_lvl_upgoing_matches_tuples = []
            for group_node in group_nodes:
                group_node_match = local_graph.get_attributes_of_node(group_node)["match"]
                group_node_score = local_graph.get_attributes_of_node(group_node)["score_intralevel"]
                downgoing_matches_msgs = []
                downgoing_matches_tuples = []
                edges_list_msgs = []
                for edge in group_node_match:
                    edge_dict = {"origin_node" : int(edge[0]), "target_node" : int(edge[1]), "score" : group_node_score}
                    edges_list_msgs.append(edge_dict)
                if matches_msgs:
                    for match_msg in matches_msgs:
                        downgoing_matches_msgs.append(match_msg + edges_list_msgs)
                    for match_tuple in matches_tuples:
                        # self.logger.info("downgoing_matches_tuples 1 - {}".format(match_tuple))
                        downgoing_matches_tuples.append(match_tuple + list(group_node_match))
                        # self.logger.info("downgoing_matches_tuples 2 - {}".format(downgoing_matches_tuples))
                else:
                    downgoing_matches_msgs = [edges_list_msgs]
                    downgoing_matches_tuples = [list(group_node_match)]
                    # self.logger.info("downgoing_matches_tuples 3 - {}".format(downgoing_matches_tuples))
                group_node_neighbourhood_graph = match_graph.get_neighbourhood_graph(group_node)

                single_nodes = group_node_neighbourhood_graph.find_nodes_by_attrs({"type": sweeped_levels[lvl], "combination_type" : "pair"})
                single_lvl_upgoing_matches_msgs = []
                
                if len(sweeped_levels) > lvl + 1:
                    for single_node in single_nodes:
                        single_node_neighbourhood_graph = match_graph.get_neighbourhood_graph(single_node)
                        final_matches_msgs, final_matches_tuples = build_matches_msg_from_match_graph_iteration(single_node_neighbourhood_graph, lvl+1, downgoing_matches_msgs, downgoing_matches_tuples)
                        single_lvl_upgoing_matches_msgs += final_matches_msgs
                        self.logger.info("single_lvl_upgoing_matches_tuples 1 - {}".format(single_lvl_upgoing_matches_tuples))
                        single_lvl_upgoing_matches_tuples += final_matches_tuples
                        self.logger.info("single_lvl_upgoing_matches_tuples 2 - {}".format(single_lvl_upgoing_matches_tuples))
                        # single_lvl_upgoing_matches_tuples = [[]]
                        # for final_match_tuples in final_matches_tuples:
                        #     for single_lvl_upgoing_match_tuples in single_lvl_upgoing_matches_tuples:
                        #         single_lvl_upgoing_matches_tuples.append(list(set(single_lvl_upgoing_match_tuples).union(set(final_match_tuples))))
                        

                else:
                    single_lvl_upgoing_matches_msgs = downgoing_matches_msgs
                    single_lvl_upgoing_matches_tuples = downgoing_matches_tuples
                    self.logger.info("single_lvl_upgoing_matches_tuples 3 - {}".format(len(single_lvl_upgoing_matches_tuples)))

                group_lvl_upgoing_matches_msgs += single_lvl_upgoing_matches_msgs
                self.logger.info("group_lvl_upgoing_matches_tuples 1 - {}".format(group_lvl_upgoing_matches_tuples))
                group_lvl_upgoing_matches_tuples += single_lvl_upgoing_matches_tuples
                self.logger.info("group_lvl_upgoing_matches_tuples 2 - {}".format(group_lvl_upgoing_matches_tuples))
            return group_lvl_upgoing_matches_msgs, group_lvl_upgoing_matches_tuples

        final_matches_msgs, final_matches_tuples = build_matches_msg_from_match_graph_iteration(match_graph, 0, [], [])
        return final_matches_msgs, final_matches_tuples


    def full_graph_clipper_filter(self, G1, G2, basis_level, basis_level_dt, matches_msg_list, matches_tuples_list):
        G1_nodes = G1.graph.nodes(data=True)
        clipper_matches_msg = []
        for i, match_tuples in enumerate(matches_tuples_list):
            basis_level_match = [pair for pair in match_tuples if G1_nodes[pair[0]]["type"] == basis_level]
            if basis_level_match:
                data1, data2, basis_level_match_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, basis_level_match, "Geometric_info")
                clipper = Clipper(basis_level_dt)
                clipper.score_pairwise_consistency(data1, data2, basis_level_match_numerical)
                clipper_match_numerical, score = clipper.solve_clipper()
                clipper_match_categorical = set(clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2))

                if score > INTERLEVEL_CLIPPER_THR and len(clipper_match_numerical) == len(basis_level_match) and\
                                                            clipper_match_categorical not in clipper_match_categorical:
                    clipper_matches_msg += [(score, matches_msg_list[i])]
        clipper_matches_msg_sorted = [clipper_matches_msg[i] for i in np.argsort([match[0] for match in clipper_matches_msg])]
        return clipper_matches_msg_sorted
