import numpy as np
import itertools
import time
import copy
import json
import os
import pathlib, sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import transforms3d.euler as eul

graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
sys.path.append(graph_matching_dir)
from graph_matching.Clipper import Clipper
from graph_matching.utils import transform_plane_definition, multilist_combinations, flatten_graph

graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
sys.path.append(graph_wrapper_dir)
from situational_graphs_wrapper.GraphWrapper import GraphWrapper


class GraphMatcher():
    def __init__(self, logger, log_level = 0):
        self.graphs = {}
        self.logger = logger
        self.log_level = log_level
        self.stored_match_graph = None
        self.stored_match_graph_dev = GraphWrapper({"nodes":[(0, {"match":set(),"split_match":[],"split_scores":[], "type" : "Plane", "merge_lvl": 0, "score_intralevel": 0})], "edges":[], "name": "deviations"})
        self.stored_consistent_combinations = []
        self.room_string = "Finite Room"
        self.ws_string = "Plane"

    def set_parameters(self, params):
        self.params = params

    def set_graph_from_dict(self, graph_def, graph_name):
        self.graphs[graph_name] = GraphWrapper(graph_def = graph_def)

    def set_graph_from_wrapper(self, graph_wrapper, graph_name):
        self.graphs[graph_name] = graph_wrapper

###  The match function performs a detailed, multi-level graph matching operation between two graphs.
    def match(self, G1_name, G2_name, add_deviations = False):
        unique_match_found = False
        if self.log_level > 0:
            self.logger.info("BENNINGING match")
        print("Add deviations flag:", add_deviations)

        start_time = time.time()
        ### Retrieve the levels to be processed from the parameters.
        swept_levels = self.params["levels"]["name"] 
        lvl = 0
        G1_full = copy.deepcopy(self.graphs[G1_name])
        G2_full = copy.deepcopy(self.graphs[G2_name])

        # if self.log_level > 9:
        #     self.plot_geometry_graphs([G1_full, G2_full], swept_levels)

        ### Initialize an empty match_graph to store the matching results using GraphWrapper.
        match_graph = GraphWrapper(graph_def={'name': "match",'nodes' : [], 'edges' : []}) 
        ###  Create a deep copy of the stored_match_graph to store the matching results.
        if self.stored_match_graph:
            stored_match_graph = copy.deepcopy(self.stored_match_graph)
            merged_stored_match_graph = stored_match_graph.filter_graph_by_node_attributes({"merge_lvl":len(swept_levels)-1, "type" : swept_levels[0]})
            filter_out_nodes = [[],[]]
            ### Extract nodes to be filtered out based on previous matches.
            for match_node_attrs in merged_stored_match_graph.get_attributes_of_all_nodes():
                filter_out_nodes[0] += list(np.array(list(match_node_attrs[1]['match']))[:,0])
                filter_out_nodes[1] += list(np.array(list(match_node_attrs[1]['match']))[:,1])
            ### Filter G1_full and G2_full to exclude previously matched nodes.
            filter_in_nodes = [list(set(list(G1_full.get_nodes_ids())) - set(filter_out_nodes[0])), list(set(list(G2_full.get_nodes_ids())) - set(filter_out_nodes[1]))]
            G1_full = G1_full.filter_graph_by_node_list(filter_in_nodes[0])
            G2_full = G2_full.filter_graph_by_node_list(filter_in_nodes[1])

        else:
            stored_match_graph = None
        ### Define a nested function match_iteration to perform the actual matching at each level.    
        def match_iteration(working_node_ID, lvl):
            print("=============================================================")
            print(f"************ INTERLEVEL CANDIDATES GENERATION - LEVEL {swept_levels[lvl]} ************")
            print("=============================================================")
            ### INTERLEVEL CANDIDATES GENERATION
            if working_node_ID:
                ### Extract every children of the parent higher-level match which belongs to current level
                working_node_attrs = match_graph.get_attributes_of_node(working_node_ID)
                self.logger.info(f"flag working_node_ID match {working_node_attrs['match']}")
                G1_lvl = G1_full.get_neighbourhood_graph(working_node_attrs["match"][0]).filter_graph_by_node_types(swept_levels[lvl])
                G2_lvl = G2_full.get_neighbourhood_graph(working_node_attrs["match"][1]).filter_graph_by_node_types(swept_levels[lvl])

            else:
                ### Extract every node in the whole graph which belongs to the current level
                G1_lvl = G1_full.filter_graph_by_node_types(swept_levels[lvl])
                G2_lvl = G2_full.filter_graph_by_node_types(swept_levels[lvl])

            ### If a stored match graph exists, extract the best pair of nodes from the previous matches and transform their geometric information for the current level.
            if self.stored_match_graph:
                stored_match_graph = copy.deepcopy(self.stored_match_graph)
                best_pair_ids = stored_match_graph.get_attributes_of_node(stored_match_graph.find_nodes_by_attrs({"type": swept_levels[-1], "combination_type" : "group","merge_lvl": 1})[0])["best_pair"]
                flag_match = stored_match_graph.get_attributes_of_node(stored_match_graph.find_nodes_by_attrs({"type": swept_levels[-1], "combination_type" : "group","merge_lvl": 1})[0])["match"]
                best_pair_attrs = [copy.deepcopy(self.graphs[G1_name].get_attributes_of_node(best_pair_ids[0])), copy.deepcopy(self.graphs[G2_name].get_attributes_of_node(best_pair_ids[1]))]
                best_pair_attrs[1]["Geometric_info"] = self.change_pos_dt(self.graphs[G2_name], [best_pair_ids[1]], self.params["levels"]["datatype"][swept_levels[-1]], self.params["levels"]["datatype"][swept_levels[lvl]])[0]
                best_pair_attrs[0]["Geometric_info"] = self.change_pos_dt(self.graphs[G1_name], [best_pair_ids[0]], self.params["levels"]["datatype"][swept_levels[-1]], self.params["levels"]["datatype"][swept_levels[lvl]])[0]
            else:
                stored_match_graph = None
            ### Compute all possible node combinations between both subGraphs
            all_pairs_categorical = set(itertools.product(G1_lvl.graph.nodes(), G2_lvl.get_nodes_ids()))
            # all_pairs_categorical = self.filter_local_match_with_global(all_pairs_categorical, full_graph_matches) # TODO include

            # TODO(dps): implementing objects filtering
            print("LEVEL: ", swept_levels[lvl])
            # categories = {}
            print(f"Number of original pairs: {len(all_pairs_categorical)}")
            print(f"Original pairs {all_pairs_categorical}")
            print("FILTERING BY CONTENT SWEEPING LEVEL: ", swept_levels[lvl])

            ### FILTERING ROOMS BY CONTENT
            filtered_pairs_by_content =  []
            if all_pairs_categorical and swept_levels[lvl] == self.room_string:
                # print(f"FILTERING BY CONTENT - {self.room_string}")
                filtered_pairs_by_content += self.filter_by_content(all_pairs_categorical, G1_full, G2_full, G1_lvl, G2_lvl)
                print(f"Room - Number of all_pairs_categorical, filtered_pairs_by_content: {len(all_pairs_categorical),  len(filtered_pairs_by_content)}")
            ## FILTERING WS BY CONTENT
            if all_pairs_categorical and swept_levels[lvl] == self.ws_string:
                # print(f"FILTERING BY CONTENT - {self.ws_string}")
                filtered_pairs_by_content += self.filter_by_content(all_pairs_categorical, G1_full, G2_full, G1_lvl, G2_lvl)
                print(f"WS - Number of all_pairs_categorical, filtered_pairs_by_content: {len(all_pairs_categorical),  len(filtered_pairs_by_content)}")


            for pair in filtered_pairs_by_content:
                if pair in all_pairs_categorical:
                    all_pairs_categorical.remove(pair)

            print(f"Number of pairs after removing objects: {len(all_pairs_categorical)}")
            print(f"Pairs after removing objects {all_pairs_categorical}")


            ### Compute all possible node combinations between the subgraphs, assess geometric consistency using the Clipper class, and filter out bad pairs.
            filtered_bad_pairs_categorical = set()
            if all_pairs_categorical and (working_node_ID):# or stored_match_graph):
                ### Assess GC of each candidate pair with higher-level parent
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, all_pairs_categorical, "Geometric_info")
                # print("GENERATED CLIPPER INPUT")
                # print(f"data1 {data1}")
                # print(f"data2 {data2}")
                # print(f"flag all_pairs_numerical {all_pairs_numerical}")
                # print(f"nodes1 {nodes1}")
                # print(f"nodes2 {nodes2}")
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["clipper_invariants"][swept_levels[lvl]], self.params, self.logger)
                n_extra_pairs = 0
                if working_node_ID:
                    n_extra_pairs += 1
                    data1, data2, all_pairs_and_parent_numerical, nodes1, nodes2 = self.add_parents_data(data1, data2, all_pairs_numerical, working_node_attrs["data_node1"],working_node_attrs["data_node2"], nodes1, nodes2)
                    origin_nodes_attrs = [working_node_attrs["data_node1"], working_node_attrs["data_node2"]]

                # print(f"data1 {data1}")
                # print(f"data2 {data2}")

                # if stored_match_graph:
                #     # self.logger.info(f"flag IN STORED M GRAPH")
                #     # self.logger.info(f"flag data1 {data1}")
                #     # self.logger.info(f"flag best_pair_attrs[0] {best_pair_attrs[0]}")
                #     n_extra_pairs += 1
                #     if not working_node_ID:
                #         all_pairs_and_parent_numerical = copy.deepcopy(all_pairs_numerical)
                #     data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_and_parent_numerical, best_pair_attrs[0]["Geometric_info"],best_pair_attrs[1]["Geometric_info"])
                #     origin_nodes_attrs = [best_pair_attrs[0]["Geometric_info"],best_pair_attrs[1]["Geometric_info"]]
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[lvl], origin_nodes_attrs[0]))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[lvl], origin_nodes_attrs[1]))
                
                plot_flag = False
                
                if plot_flag:
                    print(f"Plot geometry setlist for level {lvl}")
                    self.plot_geometry_setlist("INTERLEVEL", [data1, data2], self.params["levels"]["datatype"][swept_levels[lvl]], [nodes1,nodes2])
                    input("clipper input at intralevel")
                    
                if working_node_ID:
                    nodes1 = nodes1[:-1]
                    nodes2 = nodes2[:-1]

                # print("************ PROCESSING INTERLEVEL COMBINATION ************")
                # print(f"Clipper input all_pairs_categorical: {all_pairs_and_parent_numerical}")
                # print(f"Clipper input data1: {data1}")
                # print(f"Clipper input data2: {data2}")
                # print(f"Clipper input nodes1: {nodes1}")
                # print(f"Clipper input nodes2: {nodes2}")
                # print("**************************************************************")
                clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                # print("***************** SCORED PAIRS INTERLEVEL *****************")
                # print(f"dbg match_iteration M_aux {M_aux}")
                
                if n_extra_pairs == 1:
                    interlevel_scores = M_aux[:,-1][:-1]
                elif n_extra_pairs == 2:
                    interlevel_scores = np.array(M_aux[:,-2:])
                    interlevel_scores = (interlevel_scores[:,0] + interlevel_scores[:,1]) / 2
                    interlevel_scores = interlevel_scores[:-2]
                # self.logger.info(f"dbg match_iteration max interlevel_scores {max(interlevel_scores)}")
                # print(f"interlevel_scores {interlevel_scores}")
                # print(f"Number of interlevel scores: {len(interlevel_scores)}")
                # print(f"Threshold: {self.params['thresholds']['local_interlevel'][f'{swept_levels[0]} - {swept_levels[1]}'][0]}")

                # self.logger.info(f"flag interlevel_scores {interlevel_scores, len(interlevel_scores)}")
                good_pairs = interlevel_scores >= self.params["thresholds"]["local_interlevel"][f"{swept_levels[0]} - {swept_levels[1]}"][0]
                # good_pairs = interlevel_scores >= -1.0
                bad_pairs = [not elem for elem in good_pairs]
                filtered_bad_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[bad_pairs], nodes1, nodes2))
                filtered_good_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[good_pairs], nodes1, nodes2))
                interlevel_scores_dict = {list(filtered_good_pairs_categorical)[i]: interlevel_scores[good_pairs][i] for i in range(len(filtered_good_pairs_categorical))}
                # self.logger.info(f"flag filtered_bad_pairs_categorical {filtered_bad_pairs_categorical}")
                # self.logger.info(f"flag filtered_good_pairs_categorical {filtered_good_pairs_categorical}")
            else:
                interlevel_scores_dict = {list(all_pairs_categorical)[i]: 1. for i in range(len(all_pairs_categorical))}
                filtered_bad_pairs_categorical = set()

            
            # complete_matches_combinations = G1_lvl.matchByNodeType(G2_lvl)
            # self.logger.info(f"flag all_pairs_categorical {all_pairs_categorical}")

            # print("************ ALL POSSIBLE INTERLEVEL COMBINATIONS ************")
            # print(f"{G1_lvl.get_nodes_ids()}")
            # print(f"{G2_lvl.get_nodes_ids()}")
            # self.logger.info(f"flag G1_lvl.matchByNodeType(G2_lvl) {G1_lvl.matchByNodeType(G2_lvl)}")

            # interlevel_consistent_combinations = self.delete_list_if_element_inside(G1_lvl.matchByNodeType(G2_lvl), filtered_bad_pairs_categorical)
            # print("STANDARD INTERLEVEL COMBINATIONS GENERATION")
            # TODO(dps): implementing objects filtering
            filtered_pairs_by_content = set(filtered_pairs_by_content)
            filtered_bad_pairs_categorical.update(filtered_pairs_by_content)
            interlevel_consistent_combinations = self.remove_bad_pairs(G1_lvl.matchByNodeType(G2_lvl), filtered_bad_pairs_categorical, swept_levels[lvl], keep_length = True)
            filter1_scores = []
            filter1_matches = []
            filter1_lengths = []

            print("************ INTERLEVEL CONSISTENT COMBINATIONS ************")
            print(f"Interlevel consistent combinations")
            for interlevel_consistent_combination in interlevel_consistent_combinations:
                print(f"{list(interlevel_consistent_combination)}")
            print("**************************************************************")

            # find max length
            # max_len = max((len(c) for c in interlevel_consistent_combinations), default=0)
            max_len = len(G2_lvl.get_nodes_ids())
            print(f"MAX INTERLEVEL COMBINATION LENGTH: {max_len}")
            
            # keep only those with max length
            interlevel_consistent_combinations = frozenset(
                c for c in interlevel_consistent_combinations if len(c) == max_len
            )

            # ### !DEBUG: MATCHING FOR JUST ONE ROOM
            # if swept_levels[lvl] == self.room_string:
            #     # remove if not '72' in any of the Pairs
            #     interlevel_consistent_combinations = frozenset(
            #         c for c in interlevel_consistent_combinations if any('72' in pair for pair in c)
            #     )

            print("****** MAX INTERLEVEL CONSISTENT COMBINATIONS ******")
            print(f"Interlevel consistent combinations: {interlevel_consistent_combinations}")
            print("***************************************************")

            ### INTRALEVEL CANDIDATES COMBINATION
            ### Evaluate the consistency of candidate pairs within the same level and retain good matches. Adds these good matches to match_graph as nodes and edges
            print("=============================================================")
            print(f"************ INTRALEVEL CANDIDATES EVALUATION - LEVEL {swept_levels[lvl]} ************")
            print("=============================================================")

            # PREPARE INTRALEVEL COMPARISON PLOT
            # n_pairs = len(interlevel_consistent_combinations)
            # if n_pairs == 0:
            #     return
            # fig = plt.figure("INTRALEVEL EVALUATION", figsize=(50, 50 * n_pairs))
            # outer_gs = GridSpec(1, n_pairs, figure=fig, wspace=0.5)
            # pair_i = 0
            # plt.ion()

            for A_categorical in interlevel_consistent_combinations:
                # self.logger.info(f"flag A_categorical {A_categorical, len(A_categorical)}")
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")

                if working_node_ID:
                    ### ADD FLOOR ORIENTATION
                    floor1 = copy.deepcopy(data1[A_numerical[0][0]])
                    floor1[2:] = np.array([2,0,0,1])
                    floor2 = copy.deepcopy(data2[A_numerical[0][1]])
                    floor2[2:] = np.array([2,0,0,1])
                    data1, data2, all_pairs_and_parent_numerical, nodes1, nodes2  = self.add_parents_data(data1, data2, A_numerical, floor1,floor2,nodes1, nodes2)
                    parent_pair = all_pairs_and_parent_numerical[-1]
                    A_numerical = all_pairs_and_parent_numerical
                    ### END
                    data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[lvl], working_node_attrs["data_node1"]))
                    data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[lvl], working_node_attrs["data_node2"]))
                
                ### DEBUGGING
                # dbg_scores_list,dbg_lengths_list, dbg_tuples_list, C_list, M_list = [],[], [], [], []

                # for i in range(1):
                #     clipper = Clipper(self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["clipper_invariants"][swept_levels[lvl]], self.params, self.logger)
                #     C, M = clipper.score_pairwise_consistency(data1, data2, A_numerical)
                #     C_list.append(C)
                #     M_list.append(M)
                #     M_aux, _ = clipper.get_M_C_matrices()
                #     clipper_match_numerical, score = clipper.solve_clipper()
                #     # print("DBG BUILDING RAW ITERATION")
                #     # print(f"Score: {score}")
                #     # print(f"dbg match_iteration M_aux {M_aux}")
                #     # self.logger.info(f"dbg building raw - clipper_match_numerical {clipper_match_numerical}")
                #     clipper_match_numerical_tuples = [tuple(pair) for pair in clipper_match_numerical]

                #     dbg_scores_list.append(score)
                #     dbg_lengths_list.append(len(clipper_match_numerical))
                #     dbg_tuples_list.append(clipper_match_numerical_tuples)
                    
                # C_all_equal = all(np.array_equal(lst, C_list[0]) for lst in C_list)
                # # self.logger.info(f"dbg building raw - C_all_equal {C_all_equal}")
                # M_all_equal = all(np.array_equal(lst, M_list[0]) for lst in M_list)
                # # self.logger.info(f"dbg building raw - M_all_equal {M_all_equal}")
                
                # self.logger.info(f"dbg building raw dbg_scores_list {dbg_scores_list}")
                # self.logger.info(f"dbg building raw - dbg_scores_list equal {all(element == dbg_scores_list[0] for element in dbg_scores_list)}")
                # self.logger.info(f"dbg building raw - dbg_lengths_list equal {all(element == dbg_lengths_list[0] for element in dbg_lengths_list)}")
                # self.logger.info(f"dbg building raw - dbg_tuples_list {dbg_tuples_list}")
                # tuples_all_equal = all(set(lst) == set(dbg_tuples_list[0]) for lst in dbg_tuples_list)
                # self.logger.info(f"dbg building raw - dbg_tuples_list equal {tuples_all_equal}")
                # self.logger.info(f"dbg building raw - C_list {C_list}")
                # self.logger.info(f"dbg building raw - M_list {M_list}")

                

                plot_graph_2 = False
                if plot_graph_2:
                    self.plot_geometry_setlist(f"INTRALEVEL {A_categorical}", [data1, data2], self.params["levels"]["datatype"][swept_levels[lvl]], [nodes1, nodes2])
                    # self.generate_comparison_plots(outer_gs, fig, pair_i, data1, data2, A_categorical, swept_levels, lvl, tags)
                    # pair_i += 1
                    input("clipper input at intralevel")


                ### ORIGINAL
                clipper = Clipper(self.params["levels"]["datatype"]
                                  [swept_levels[lvl]], self.params["levels"]
                                  ["clipper_invariants"][swept_levels[lvl]], self.params, self.logger)

                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                clipper_match_numerical, score = clipper.solve_clipper()
                ### END

                # ### ADD FLOOR ORIENTATION
                # if working_node_ID: 
                #     # self.logger.info(f"FLAG clipper_match_numerical {clipper_match_numerical}")
                #     # self.logger.info(f"FLAG M_aux {M_aux}")
                #     index = -1
                #     for i, e in enumerate(clipper_match_numerical):
                #         if np.array_equal(e, parent_pair):
                #             index = i
                #             break
                #     if index != -1:
                #         clipper_match_numerical = np.delete(clipper_match_numerical, index, axis= 0)
                #     else:
                #         score = 0.
                # ### END
                print("************ PROCESSING INTRALEVEL COMBINATION ************")
                print(f"Clipper input A_numerical: {A_numerical}")
                print(f"Clipper input A_categorical: {A_categorical}")
                print(f"Clipper input data1: {data1}")
                print(f"Clipper input data2: {data2}")
                print(f"Clipper input nodes1: {nodes1}")
                print(f"Clipper input nodes2: {nodes2}")
                # print(f"Clipper input A_numerical: {A_numerical}")
                # print(f"Clipper input nodes1: {nodes1}")
                # print(f"Clipper input nodes2: {nodes2}")
                # print(f"Clipper output match numerical: {clipper_match_numerical}")
                print(f"Clipper output score: {score}")
                clipper_match_categorical = set(clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2))
                print(f"Clipper output match categorical: {clipper_match_categorical}")

                if working_node_ID:
                    nodes1 = nodes1[:-1]
                    nodes2 = nodes2[:-1]
                    unparented_clipper_match_categorical = []
                    for clipper_match_categorical_pair in clipper_match_categorical:
                        if clipper_match_categorical_pair[0] != 'parent' and clipper_match_categorical_pair[1] != 'parent':
                            unparented_clipper_match_categorical.append(clipper_match_categorical_pair)
                    clipper_match_categorical = unparented_clipper_match_categorical

                # self.logger.info(f"dbg match_iteration clipper_match_categorical {clipper_match_categorical}")
                print("**************************************************************")
                # floor_condition = self.assess_floor_consistency(data1, data2, swept_levels[lvl])
                floor_condition = True
                if working_node_ID:
                    floor_condition = self.assess_floor_consistency(data1, data2, swept_levels[lvl], A_numerical)
                else:
                    floor_condition = True
                print(f"FLOOR CONDITION: {floor_condition}")


                if score > self.params["thresholds"]["local_intralevel"][swept_levels[lvl]][0] and clipper_match_categorical not in filter1_matches and floor_condition:
                    print("********************************")
                    print("GOOD INTRALEVEL MATCH FOUND")
                    print(f"Match: {clipper_match_categorical}")
                    print(f"Score: {score}")
                    print("********************************")
                    filter1_scores.append(score)
                    filter1_matches.append(clipper_match_categorical)
                    filter1_lengths.append(len(clipper_match_categorical))
            # plt.tight_layout()
            plt.show(block=False)
            # plt.pause(0.1)
            # plt.show(block=True)
            ### Add good submatches to the match_graph.      
            if filter1_scores:
                print("=============================================================")
                print(f"************ ADDING GOOD INTRALEVEL MATCHES TO MATCH GRAPH - LEVEL {swept_levels[lvl]} ************")
                print("=============================================================")
                print(f"filter1_matches: {filter1_matches}")
                print(f"filter1_scores: {filter1_scores}")
                sorted_matches_indexes = [index for index, val in enumerate(filter1_lengths) if val == max(filter1_lengths)]
                print(f"sorted_matches_indexes: {sorted_matches_indexes}")

                if working_node_ID:
                    best_submatch_score = max([filter1_scores[i] for i in sorted_matches_indexes])
                    match_graph.update_node_attrs(working_node_ID, {"downstream_score" : best_submatch_score})
                    print(f"Updated working node {working_node_ID} with downstream_score {best_submatch_score}")

                print("************ ADDING TO MATCH GRAPH ************")
                print("MATCH GRAPH BEFORE ADDING:")
                print(match_graph.graph)

                # sorted_matches_indexes = range(len(filter1_lengths))
                for good_submatch_i in sorted_matches_indexes:
                    group_node_id = match_graph.get_total_number_nodes() + 1
                    node_attr = [(group_node_id, {"type": swept_levels[lvl], "match": filter1_matches[good_submatch_i], "merge_lvl" :0,\
                                    "combination_type" : "group", "score_intralevel" : filter1_scores[good_submatch_i], "best_pair" : False})]
                    if working_node_ID:
                        edges_attr = [(working_node_ID, group_node_id, {})]
                    else:
                        edges_attr = []
                    match_graph.add_subgraph(node_attr, edges_attr)


                    ### Next level
                    ### Recursively handle matching at the next level by transforming node data and calling match_iteration on the next level.
                    if lvl < len(swept_levels) - 1:
                        parent1_data = self.change_pos_dt(G1_full, np.array(list(filter1_matches[good_submatch_i]))[:,0], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl+1]])
                        parent2_data = self.change_pos_dt(G2_full, np.array(list(filter1_matches[good_submatch_i]))[:,1], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl+1]])
                    else: 
                        parent1_data = self.change_pos_dt(G1_full, np.array(list(filter1_matches[good_submatch_i]))[:,0], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl]])
                        parent2_data = self.change_pos_dt(G2_full, np.array(list(filter1_matches[good_submatch_i]))[:,1], self.params["levels"]["datatype"][swept_levels[lvl]], self.params["levels"]["datatype"][swept_levels[lvl]])
                   

                    for i, lowerlevel_match_pair in enumerate(filter1_matches[good_submatch_i]):
                        existing_nodes = match_graph.find_nodes_by_attrs({"type": swept_levels[lvl], \
                                            "match" : lowerlevel_match_pair, "combination_type" : "pair"})
                        if not existing_nodes:

                            # G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match_pair[0])
                            # G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match_pair[1])

                            pair_node_id = match_graph.get_total_number_nodes() + 1
                            node_attr = [(pair_node_id, {"type": swept_levels[lvl], "match": lowerlevel_match_pair,\
                                                        "combination_type" : "pair", "score_interlevel" : interlevel_scores_dict[lowerlevel_match_pair],\
                                                        "data_node1" : parent1_data[i], "data_node2" : parent2_data[i], "merge_lvl" :0, "best_pair" : False})]

                            edges_attr = [(pair_node_id, group_node_id, {})]

                            match_graph.add_subgraph(node_attr, edges_attr)

                            # next_parents_data = {"match" : lowerlevel_match_pair, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : pair_node_id}
                            # match_iteration(pair_node_id, lvl + 1, next_parents_data)
                            if lvl < len(swept_levels) - 1:
                                match_iteration(pair_node_id, lvl + 1)

                        else:
                            edges_attr = [(existing_nodes[0], group_node_id, {})]
                            match_graph.add_edges(edges_attr)

            elif working_node_ID:
                match_graph.update_node_attrs(working_node_ID, {"downstream_score" : 0.0})

            print("MATCH GRAPH AFTER ADDING:")
            print(match_graph.graph)

            ### Prune the match_graph to remove inconsistent matches and select the best localization pair for the next level.
            if lvl < len(swept_levels) - 1:
                if self.log_level > 0:
                    self.draw_as_match_graph(match_graph, "raw match graph")
                n_nodes_raw_match_graph = len(match_graph.get_nodes_ids())
                # self.logger.info(f"dbg n_nodes_raw_match_graph {n_nodes_raw_match_graph}")
                self.prune_interlevel(match_graph, self.graphs[G1_name], self.graphs[G2_name], swept_levels[lvl:lvl+2])
                self.select_best_global_localization_pair(match_graph, swept_levels[lvl:lvl+2])
                # self.add_upranted_nodes_by_level(match_graph, G1_full, G2_full, swept_levels[lvl:lvl+2])
                print(f"PRUNED MATCH GRAPH AT LEVEL {swept_levels[lvl]}:")
                print(match_graph.graph)

            print("=============================================================")

            return
        ### End match_iteration function.

        ### Check if there are nodes in both graphs, then call match_iteration to start the matching process from the top level. Visualize the match_graph and gather the final combinations of matched nodes.
        if G1_full.get_nodes_ids() and G2_full.get_nodes_ids():

            match_iteration(None, lvl)
            if len(list(match_graph.get_nodes_ids())) != 0 and self.log_level > 0:
                self.draw_as_match_graph(match_graph, "match graph")

            if add_deviations:
                self.add_deviated_nodes_by_level(match_graph, G1_full, G2_full, swept_levels[lvl:lvl+2])
            final_combinations = self.gather_final_combinations_from_match_graph(G1_full, G2_full, match_graph, swept_levels)

        else:
            final_combinations = []

        print(f"FINAL COMBINATIONS: {len(final_combinations)}")
        # print(f"{final_combinations}")
        max_combination_length = 0
        for comb in final_combinations:
            # print(comb)
            if len(comb) > max_combination_length:
                max_combination_length = len(comb)
        print(f'{max_combination_length=}')
        filtered_list = []
        for comb in final_combinations:
            if len(comb) < max_combination_length:
                print("Removing combination due to length inconsistency with length: ", len(comb))
                filtered_list.append(comb)
        for item in filtered_list:
            final_combinations.remove(item)
        print(f"FINAL COMBINATIONS AFTER LENGTH FILTERING: {len(final_combinations)}")
        print(f"final_combinations:{final_combinations}")

        ###  Log the number of good matches found. If only one match is found, update the stored match graph with the current match. Handle cases with multiple symmetries by logging the scores of the matches. Return a success flag and the final combinations of matches.
        if final_combinations:
            self.logger.info("Found {} good matches!!!".format(len(final_combinations)))
            success = True

            if len(final_combinations) == 1:
                self.logger.info("Only one match succeded with score - {}".format(final_combinations[0][0]["score"]))
                unique_match_found = True

                if not self.stored_match_graph:
                    self.stored_match_graph = match_graph.filter_graph_by_node_attributes({"merge_lvl":1})

                else:
                    for swept_level in swept_levels:
                        current_node_id = list(match_graph.filter_graph_by_node_attributes({"merge_lvl":1, "type" : swept_level}).get_nodes_ids())[0]
                        current_node_attrs = match_graph.get_attributes_of_node(current_node_id)
                        stored_node_id = list(self.stored_match_graph.filter_graph_by_node_attributes({"merge_lvl":1, "type" : swept_level}).get_nodes_ids())[0]
                        stored_node_attrs = self.stored_match_graph.get_attributes_of_node(stored_node_id)
                        stored_node_attrs["match"].update(current_node_attrs["match"])
                        stored_node_attrs["score_intralevel"] = current_node_attrs["score_intralevel"]
                        if "split_match" in stored_node_attrs.keys():
                            stored_node_attrs['split_match'] += current_node_attrs['split_match']
                            stored_node_attrs['split_scores'] += current_node_attrs['split_scores']

                    # node_color = self.stored_match_graph.define_draw_color_option_by_node_type()
                    # node_size = self.stored_match_graph.define_node_size_option_by_combination_type_attr()
                    # linewidths = self.stored_match_graph.define_node_linewidth_option_by_combination_type_attr()
                    # options = {'node_color': node_color, 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    #             "node_size" : node_size, "linewidths" : linewidths, "edgecolors" : "black"}
                    # self.stored_match_graph.draw("test stored match graph", options = options, show = self.params["verbose"])

                    # node_color = match_graph.define_draw_color_option_by_node_type()
                    # node_size = match_graph.define_node_size_option_by_combination_type_attr()
                    # linewidths = match_graph.define_node_linewidth_option_by_combination_type_attr()
                    # options = {'node_color': node_color, 'node_size': 50, 'width': 2, 'with_labels' : True,\
                    #             "node_size" : node_size, "linewidths" : linewidths, "edgecolors" : "black"}
                    # match_graph.draw("test current graph", options = options, show = self.params["verbose"])
                    # time.sleep(999)
                    
                final_combinations_full = self.gather_final_combinations_from_match_graph(self.graphs[G1_name], self.graphs[G2_name], self.stored_match_graph, swept_levels)
                # self.logger.info(f"flag to start function for dev")
                final_combinations_dev = self.gather_final_combinations_from_match_graph(self.graphs[G1_name], self.graphs[G2_name], self.stored_match_graph_dev, [swept_levels[-1]])
                # self.logger.info(f"flag final_combinations_dev {final_combinations_dev}")

            elif len(final_combinations) > 1:
                final_combinations_full = []
                final_combinations_dev = []
                # self.logger.info("{} symmetries detected. Scores - {}".format(len(final_combinations), [match[0]["score"] for match in final_combinations]))

            if self.log_level > 0:
                self.subplots_match(G1_name, G2_name, final_combinations)

        else:
            success = False
            final_combinations = []
            final_combinations_full = []
            final_combinations_dev = []

        # self.logger.info("Elapsed time in match {}".format(time.time() - start_time))
        ###  Return a tuple containing the success flag, final combinations of matches, full matches, and deviated matches.
        print(f'{final_combinations_dev=}')
        return(success, final_combinations, final_combinations_full, final_combinations_dev)


    def filter_local_match_with_global(self, local_match, global_matches):
        filtered = set([ local_elem for local_elem in local_match if any(local_elem in global_match for global_match in global_matches)])
        return filtered


    def check_match_not_in_list(self, new_match, other_matches):
        if not other_matches:
            return True
        else:
            return( not any(set([tuple(pair) for pair in new_match]) == set([tuple(pair) for pair in match]) for match in other_matches))


    # def generate_clipper_input(self, G1_in, G2_in, A_categorical, feature_name):
    #     G1 = copy.deepcopy(G1_in)
    #     G2 = copy.deepcopy(G2_in)
    #     nodes1, nodes2 = list(np.array(list(A_categorical))[:,0]), list(np.array(list(A_categorical))[:,1])
    #     data1 = G1.stack_nodes_feature(nodes1, feature_name)
    #     data2 = G2.stack_nodes_feature(nodes2, feature_name)
    #     A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
    #     return(data1, data2, A_numerical, nodes1, nodes2)

    def generate_clipper_input(self, G1_in, G2_in, A_categorical, feature_name):
        G1 = copy.deepcopy(G1_in)
        G2 = copy.deepcopy(G2_in)
        nodes1, nodes2 = list(set(np.array(list(A_categorical))[:,0])), list(set(np.array(list(A_categorical))[:,1]))
        # print(f"nodes1 {nodes1}")
        # print(f"nodes2 {nodes2}")
        data1 = G1.stack_nodes_feature(nodes1, feature_name)
        data2 = G2.stack_nodes_feature(nodes2, feature_name)
        A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
        return(data1, data2, A_numerical, nodes1, nodes2)


    def change_pos_dt(self, graph_in, node_list, in_dt, out_dt):
        graph = copy.deepcopy(graph_in)
        original_aux = graph.stack_nodes_feature(node_list, "Geometric_info")
        original = copy.deepcopy(original_aux)
        if in_dt == out_dt:
            processed = original
        elif in_dt == "points" and out_dt == "points&normal":
            normal = np.repeat(np.array([[0.,0.,1.]]), len(original), axis= 0)
            processed = np.concatenate((original, normal),axis=1)
        elif in_dt == "points&normal" and out_dt == "points":
            processed = copy.deepcopy(original[:, :3])

        return(processed)


    def add_parents_data(self, data1, data2, A_numerical, data_parent1, data_parent2, nodes1, nodes2):
        # print("DEBUG ***************** ADDING PARENT DATA *****************")
        # print(f"data1 before {data1}")
        # print(f"data2 before {data2}")
        if len(np.array(data_parent1).shape) == 1:
            # data_parent1[2] = data_parent1[2] + 1.0
            # data_parent2[2] = data_parent2[2] + 1.0
            data_parent1 = [data_parent1]
            data_parent2 = [data_parent2]
            A_numerical_with_parent = np.concatenate((A_numerical, [[data1.shape[0], data2.shape[0]]]), axis= 0, dtype = np.int32)
        else:
            # for p_data in data_parent1:
            #     p_data[2] = p_data[2] + 1.0
            # for p_data in data_parent2:
            #     p_data[2] = p_data[2] + 1.0
            aux = np.dstack((np.arange(data1.shape[0], data1.shape[0] + data_parent1.shape[0]), np.arange(data1.shape[0], data1.shape[0] + data_parent1.shape[0])))[0]
            A_numerical_with_parent = np.concatenate((A_numerical, aux), axis= 0, dtype = np.int32)

        # print(f"data_parent1 after {data_parent1}")
        # print(f"data_parent2 after {data_parent2}")
            
        data1 = np.concatenate((data1, data_parent1), axis= 0, dtype = np.float64)
        data2 = np.concatenate((data2, data_parent2), axis= 0, dtype = np.float64)

        nodes1 = copy.deepcopy(nodes1)
        nodes2 = copy.deepcopy(nodes2)
        nodes1.append("parent")
        nodes2.append("parent")
        # print(f"data1 after {data1}")
        # print(f"data2 after {data2}")
        # print("DEBUG *******************************************************")
        return(data1, data2, A_numerical_with_parent, nodes1, nodes2)


    def add_floor_data(self, data1, data2, A_numerical):
        floor_pair_numerical = [data1.shape[0], data2.shape[0]]
        floor_points = [data1[A_numerical[0][0]], data2[A_numerical[0][1]]]
        A_numerical_with_parent = np.concatenate((A_numerical, [floor_pair_numerical]), axis= 0, dtype = np.int32)
        data1 = np.concatenate(([ data1, [[floor_points[0][0],floor_points[0][1],floor_points[0][2],0,0,1]]]), axis= 0, dtype = np.float64)
        data2 = np.concatenate(([ data2, [[floor_points[0][0],floor_points[0][1],floor_points[0][2],0,0,1]]]), axis= 0, dtype = np.float64)
        return(data1, data2, A_numerical_with_parent, floor_pair_numerical)

    def assess_floor_consistency(self, data1, data2, merged_level, A_numerical):
        def compute_transformation(points_a, normals_a, points_b, normals_b):
            # Compute the centroids of both sets
            centroid_a = np.mean(points_a, axis=0)
            centroid_b = np.mean(points_b, axis=0)

            # Translate points to align centroids with the origin
            points_a_centered = points_a - centroid_a
            points_b_centered = points_b - centroid_b

            # Compute the optimal rotation matrix using Singular Value Decomposition (SVD)
            H = np.dot(points_a_centered.T, points_b_centered)
            U, S, Vt = np.linalg.svd(H)
            rotation_matrix = np.dot(Vt.T, U.T)

            # Ensure the rotation matrix is proper (det(rotation) should be 1)
            if np.linalg.det(rotation_matrix) < 0:
                Vt[2, :] *= -1
                rotation_matrix = np.dot(Vt.T, U.T)

            # Apply the rotation matrix to the normals as well
            normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

            # Check for reflection by comparing normals
            reflection_needed = False
            for normal_a_transformed, normal_b in zip(normals_a_transformed, normals_b):
                if np.dot(normal_a_transformed, normal_b) < 0:
                    reflection_needed = True
                    break

            # If reflection is needed, apply it to the rotation matrix
            if reflection_needed:
                reflection_matrix = np.diag([1, 1, -1])
                rotation_matrix = np.dot(rotation_matrix, reflection_matrix)
                normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

            # Compute the translation vector
            translation_vector = centroid_b - np.dot(centroid_a, rotation_matrix.T)

            return translation_vector, rotation_matrix, reflection_needed


        # if merged_level == "Finite Room":
        #     data1 = np.concatenate(([ data1, np.tile([0,0,1], (data1.shape[0], 1))]), axis= 1, dtype = np.float64)
        #     data2 = np.concatenate(([ data2, np.tile([0,0,1], (data1.shape[0], 1))]), axis= 1, dtype = np.float64)
        # # a_all = np.concatenate(([ data1, [[0,0,0,0,0,1]]]), axis= 0, dtype = np.float64)
        # # b_all = np.concatenate(([ data2, [[0,0,0,0,0,1]]]), axis= 0, dtype = np.float64)
        
        a_all = copy.deepcopy(data1)[np.array(A_numerical)[:,0]]
        b_all = copy.deepcopy(data2)[np.array(A_numerical)[:,1]]
        # for a_all_i in copy.deepcopy(data1):
        #     new_row = [[a_all_i[0],a_all_i[1],a_all_i[2],0,0,1]]
        #     a_all = np.concatenate(([ a_all, new_row]), axis= 0, dtype = np.float64)
        # for b_all_i in copy.deepcopy(data2):
        #     new_row = [[b_all_i[0],b_all_i[1],b_all_i[2],0,0,1]]
        #     b_all = np.concatenate(([ b_all, new_row]), axis= 0, dtype = np.float64)

        points_a, normals_a = a_all[:, :3], a_all[:, -3:]
        points_b, normals_b = b_all[:, :3], b_all[:, -3:]
        translation, final_matrix, reflection_needed = compute_transformation(points_a, normals_a, points_b, normals_b)
        rotation_cond = final_matrix[0,0] > 0.8 and final_matrix[1,1] > 0.8 and abs(final_matrix[0,1]) < 0.2
        # if self.log_level > 4:
        if False:
            self.plot_geometry_setlist("floor detection", [a_all, b_all], self.params["levels"]["datatype"][merged_level])
            plt.draw()
            plt.pause(0.001)
            # print(f"dbg floor_cond {final_cond}")
            # print("Press any key to continue...")
            # key = keyboard.wait()
            # print(f"You pressed {key}")
        ####################################
        if reflection_needed:
            print("REJECTED BECAUSE REFLECTION NEEDED")
            return False
        return True
        ####################################
        # final_cond = not(reflection_needed) and rotation_cond
        

        # A_numerical_add = copy.deepcopy(A_numerical)
        # A_numerical_add[:, 0] = A_numerical_add[:, 0] + (max(A_numerical_add[:, 0])+1) * np.ones(len(A_numerical_add[:, 0]))
        # A_numerical_add[:, 1] = A_numerical_add[:, 1] + (max(A_numerical_add[:, 1])+1) * np.ones(len(A_numerical_add[:, 1]))
        # A_numerical_all = np.concatenate((A_numerical, A_numerical_add),axis=0)
        # initial_A_numerical_all_lenght = A_numerical_all.shape[0]
        # # print(f'dbg A_numerical_all {A_numerical_all}')
        # # print(f"dbg a_all {a_all}")
        # clipper = Clipper(self.params["levels"]["datatype"][merged_level], self.params["levels"]["clipper_invariants"][merged_level], self.params, self.logger)
        # clipper.score_pairwise_consistency(a_all, b_all, A_numerical_all)
        # clipper_match_numerical, score = clipper.solve_clipper()
        # match_numerical_lenght = clipper_match_numerical.shape[0]
        # # consistency_avg = clipper.get_score_all_inital_u()
        # print(f'dbg initial_A_numerical_all_lenght {initial_A_numerical_all_lenght} match_numerical_lenght {match_numerical_lenght} cond {match_numerical_lenght == initial_A_numerical_all_lenght}')
        # # print(f'dbg consistency_avg {clipper_match_numerical.shape}')
        # # print(f'dbg score {score}')
        # final_cond = match_numerical_lenght == initial_A_numerical_all_lenght

        # return final_cond

    
    # def assess_floor_consistency(self, data1, data2, merged_level):
    #     def compute_transformation(points_a, normals_a, points_b, normals_b):
    #         # Compute the centroids of both sets
    #         centroid_a = np.mean(points_a, axis=0)
    #         centroid_b = np.mean(points_b, axis=0)

    #         # Translate points to align centroids with the origin
    #         points_a_centered = points_a - centroid_a
    #         points_b_centered = points_b - centroid_b

    #         # Compute the optimal rotation matrix using Singular Value Decomposition (SVD)
    #         H = np.dot(points_a_centered.T, points_b_centered)
    #         U, S, Vt = np.linalg.svd(H)
    #         rotation_matrix = np.dot(Vt.T, U.T)

    #         # Ensure the rotation matrix is proper (det(rotation) should be 1)
    #         if np.linalg.det(rotation_matrix) < 0:
    #             Vt[2, :] *= -1
    #             rotation_matrix = np.dot(Vt.T, U.T)

    #         # Apply the rotation matrix to the normals as well
    #         normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

    #         # Check for reflection by comparing normals
    #         reflection_needed = False
    #         for normal_a_transformed, normal_b in zip(normals_a_transformed, normals_b):
    #             if np.dot(normal_a_transformed, normal_b) < 0:
    #                 reflection_needed = True
    #                 break

    #         # If reflection is needed, apply it to the rotation matrix
    #         if reflection_needed:
    #             reflection_matrix = np.diag([1, 1, -1])
    #             rotation_matrix = np.dot(rotation_matrix, reflection_matrix)
    #             normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

    #         # Compute the translation vector
    #         translation_vector = centroid_b - np.dot(centroid_a, rotation_matrix.T)

    #         return translation_vector, rotation_matrix, reflection_needed


    #     if merged_level == "Finite Room":
    #         data1 = np.concatenate(([ data1, np.tile([0,0,1], (data1.shape[0], 1))]), axis= 1, dtype = np.float64)
    #         data2 = np.concatenate(([ data2, np.tile([0,0,1], (data1.shape[0], 1))]), axis= 1, dtype = np.float64)
    #     a_all = np.concatenate(([ data1, [[0,0,0,0,0,1]]]), axis= 0, dtype = np.float64)
    #     b_all = np.concatenate(([ data2, [[0,0,0,0,0,1]]]), axis= 0, dtype = np.float64)

    #     points_a, normals_a = a_all[:, :3], a_all[:, -3:]
    #     points_b, normals_b = b_all[:, :3], b_all[:, -3:]
    #     translation, final_matrix, reflection_needed = compute_transformation(points_a, normals_a, points_b, normals_b)
    #     rotation_cond = final_matrix[0,0] > 0.8 and final_matrix[1,1] > 0.8 and abs(final_matrix[0,1]) < 0.2
    #     final_cond = not(reflection_needed) and rotation_cond
    #     print(f"DEBUG FLOOR DETECTION:")
    #     print(f"Translation: {translation}")
    #     print(f"Rotation matrix:\n{final_matrix}")
    #     print(f"Reflection needed: {reflection_needed}")
    #     print(f"Rotation condition: {rotation_cond}")
    #     print(f"Final condition: {final_cond}")
    #     if self.log_level > 4:
    #         self.plot_geometry_setlist("floor detection", [a_all, b_all], self.params["levels"]["datatype"][merged_level])
    #         plt.draw()
    #         plt.pause(0.001)
    #         # print(f"dbg floor_cond {final_cond}")
    #         # print("Press any key to continue...")
    #         # key = keyboard.wait()
    #         # print(f"You pressed {key}")

    #     return final_cond

    # def assess_floor_consistency(self, data1, data2, merged_level):

    #     def compute_transformation(points_a, normals_a, points_b, normals_b):
    #         centroid_a = points_a.mean(axis=0)
    #         centroid_b = points_b.mean(axis=0)

    #         A = points_a - centroid_a
    #         B = points_b - centroid_b

    #         # Kabsch (row-vector convention): find R s.t. A @ R ≈ B
    #         H = A.T @ B
    #         U, S, Vt = np.linalg.svd(H)
    #         R = Vt.T @ U.T

    #         # enforce proper rotation (det = +1)
    #         if np.linalg.det(R) < 0:
    #             Vt[-1, :] *= -1
    #             R = Vt.T @ U.T

    #         # normals check (row-vector convention: n' = n @ R)
    #         normals_a_rot = normals_a @ R
    #         reflection_needed = any((na @ nb) < 0 for na, nb in zip(normals_a_rot, normals_b))

    #         # Translation consistent with row-vectors: p_b ≈ p_a @ R + t
    #         t = centroid_b - (centroid_a @ R)

    #         return t, R, reflection_needed

    #     # Ensure we have Nx6: [x y z nx ny nz]
    #     if merged_level == "Finite Room":
    #         n1 = data1.shape[0]
    #         n2 = data2.shape[0]
    #         data1 = np.hstack([data1, np.tile([0.0, 0.0, 1.0], (n1, 1))]).astype(np.float64)
    #         data2 = np.hstack([data2, np.tile([0.0, 0.0, 1.0], (n2, 1))]).astype(np.float64)
    #     else:
    #         data1 = data1.astype(np.float64)
    #         data2 = data2.astype(np.float64)

    #     # Add anchor row
    #     a_all = np.vstack([data1, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    #     b_all = np.vstack([data2, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    #     points_a, normals_a = a_all[:, :3], a_all[:, 3:6]
    #     points_b, normals_b = b_all[:, :3], b_all[:, 3:6]

    #     translation, R, reflection_needed = compute_transformation(points_a, normals_a, points_b, normals_b)

    #     rotation_cond = (R[0, 0] > 0.8 and R[1, 1] > 0.8 and abs(R[0, 1]) < 0.2)
    #     # final_cond = (not reflection_needed) and rotation_cond
    #     final_cond = (not reflection_needed)

    #     print(f"DEBUG FLOOR DETECTION:")
    #     print(f"Translation: {translation}")
    #     print(f"Rotation matrix:\n{R}")
    #     print(f"Reflection needed: {reflection_needed}")
    #     print(f"Rotation condition: {rotation_cond}")
    #     print(f"Final condition: {final_cond}")

    #     if self.log_level > 4:
    #         self.plot_geometry_setlist("floor detection", [a_all, b_all], self.params["levels"]["datatype"][merged_level])
    #         plt.draw()
    #         plt.pause(0.001)

    #     return final_cond


    
    # def delete_floor_data(self, data1, data2, A_numerical):
    #     A_numerical = A_numerical[1:]
    #     self.logger.info("flag data1 {}".format(data1))
    #     data1 = data1[1:]
    #     self.logger.info("flag data1 {}".format(data1))

    def shift_plane_origin(self, n, d, p, normalize=False):
        """
        Plane: n·x + d = 0 (in the original coordinate system with origin at 0)
        New coordinate system: origin moved to p (i.e., x = x' + p)
        Returns (n, d') such that n·x' + d' = 0
        """
        n = np.asarray(n, dtype=float).reshape(3)
        p = np.asarray(p, dtype=float).reshape(3)
        d = float(d)
    
        d2 = d + n.dot(p)
    
        if normalize:
            s = np.linalg.norm(n)
            if s > 0:
                n = n / s
                d2 = d2 / s
    
        return n, d2

    def geometric_info_transformation(self, data_in, level, parent_data_in):
        # print("DEBUG ********* GEOMETRIC INFO TRANSFORMATION *********")
        # print(f"level {level}")
        # print(f"parent_data_in {parent_data_in}")
        # print(f"data_in {data_in}")
        data = copy.deepcopy(data_in)
        parent_data = copy.deepcopy(parent_data_in)
        # print(f"Transforming geometric info at level {level} with parent data {parent_data}")
        # print(f"data before transform {data}")
        if level == self.ws_string:
            # print("Applying plane transformation at level: ", level)
            for i in range(data.shape[0] - 1):
                plane = data[i]
                n = plane[3:]
                p0 = plane[:3]
        
                # plane: n·x + d = 0
                d = -p0.dot(n)
        
                # shift origin to parent position p (x = x' + p)
                n2, d2 = self.shift_plane_origin(n, d, parent_data[:3], normalize=True)
        
                data[i, 3:] = n2
                data[i, :3] = -d2 * n2   # closest point to the new origin (valid if n2 is unit)
            data[-1][:3] = data[-1][:3] - parent_data[:3]  # transform floor point
            # data[-1][2] = 5.0  # FIXME: TEST. REMOVE THIS


            # if len(parent_data) == 3:# TODO use parent dt
            #     rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])
            # elif len(parent_data) == 6:
            #     normal = parent_data[3:]
            #     psi = np.arctan2(normal[1], normal[0])
            #     rotation= eul.euler2mat(0, 0, psi, axes='sxyz')
            # else:
            #     raise ValueError("Wrong parent_data")
            # translation = -parent_data[:3]
            # print(f"translation vector {translation}")
            # print(f"rotation matrix {rotation}")

            # transformed = transform_plane_definition(data, -parent_data[:3], rotation, self.logger)
            # print(f"data after transform {data}")
        else: # TODO: translate room
            # print("No transformation applied at this level")
            transformed = data

        # print(f"data after transform {data}")
        # print("DEBUG *******************************************************")
        return data
        # return transformed


    def subplots_match(self, g1_name, g2_name, matches):
        for candidate_i, match in enumerate(matches):
            fig, axs = plt.subplots(nrows=3, ncols=2, num='Match between {} and {} graphs. Candidate {}'.format(g1_name, g2_name, candidate_i), figsize=(20, 14))
            fig.suptitle('Match between {} and {} graphs. Candidate {}'.format(g1_name, g2_name, candidate_i))

            ### Plot base graph
            plt.sca(axs[0,0])
            axs[0, 0].set_title('Base graph - {}'.format(g1_name))
            options_base = {'node_color': self.graphs[g1_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
            self.graphs[g1_name].draw(None, options_base, True)

            ### Plot target graph
            plt.sca(axs[0, 1])
            axs[0, 1].set_title('Target graph - {}'.format(g2_name))
            options_target = {'node_color': self.graphs[g2_name].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
            self.graphs[g2_name].draw(None, options_target, True)

            ### Plot base graph with match
            plt.sca(axs[1,0])
            axs[1,0].set_title('Base graph match - {}'.format(g1_name))
            nodes_base = [pair["origin_node"] for pair in match]
            options_base_matched = self.graphs[g1_name].define_draw_color_from_node_list(options_base, nodes_base, unmatched_color = None, matched_color = "grey")
            self.graphs[g1_name].draw(None, options_base_matched, True)

            ### Plot target graph with match
            plt.sca(axs[1,1])
            axs[1,1].set_title('Target graph match - {}'.format(g2_name))
            nodes_target = [pair["target_node"] for pair in match]
            options_target_matched = self.graphs[g2_name].define_draw_color_from_node_list(options_target, nodes_target, unmatched_color = None, matched_color = "grey")
            self.graphs[g2_name].draw(None, options_target_matched, True)

            ### Combined match
            plt.sca(axs[2,0])  # Use the bottom-left subplot for combined match
            ### Combined match
            plt.sca(axs[2,0])  # Use the bottom-left subplot for combined match
            match_zip = np.stack([[str(pair["origin_node"]), str(pair["target_node"])] for pair in match])
            g1_filtered = self.graphs[g1_name].filter_graph_by_node_list(match_zip[:,0])
            g2_filtered = self.graphs[g2_name].filter_graph_by_node_list(match_zip[:,1])
            mapping = {}
            node_id_diff = 100
            for last_id in list(g2_filtered.get_nodes_ids()):
                mapping[str(last_id)] = str(int(last_id) + int(node_id_diff))
            g2_filtered.relabel_nodes(mapping = mapping, copy = True)
            # g2_filtered.translate_attr_all_nodes("draw_pos", np.array([10,0]))

            combined_graph = g1_filtered.merge_graph(g2_filtered)
            match_edges_attr = []
            for pair in match:
                match_edges_attr.append((str(pair["origin_node"]), mapping[str(pair["target_node"])], {"color" : 'r'}))
            combined_graph.add_subgraph([], match_edges_attr)
            combined_graph.draw(None, None, True)
            
            plt.show(block=False)


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


    def generate_good_pairs(self, lists):
        good_pairs = set()
        for current_list in lists:
            for pair in itertools.combinations(current_list, 2):
                print(f"Generating good pair from pair {pair}")
                good_pairs.add(frozenset((pair)))
        return frozenset(good_pairs)

    def remove_bad_pairs(self, lists, bad_pairs, level, keep_length):
        lists_1 = []
        incoming_length = len(lists[0])
        for current_list in lists:
            for bad_pair in bad_pairs:
                if bad_pair in current_list:
                    current_list.remove(bad_pair)

            if current_list:
                if (level != self.room_string and len(current_list) < 2):
                    continue
                lists_1.append(frozenset(current_list))
            
        lists_2 = [current_list for current_list in lists_1 if len(current_list) == incoming_length]

        return frozenset(lists_2)
            

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

        if len(list(pruned_match_graph.get_nodes_ids())) != 0 and self.log_level > 0:
            self.draw_as_match_graph(pruned_match_graph, "pruned match graph")

        def gather_final_combinations_from_match_graph_iteration(working_node_ID, lvl):
            if working_node_ID != None:
                #### NEW
                working_node_attrs = pruned_match_graph.get_attributes_of_node(working_node_ID)
                if lvl == 1:
                    working_node_matches = working_node_attrs["match"]
                    working_node_score = working_node_attrs["score_intralevel"]
                    working_node_tuples = [{"origin_node" : int(working_node_match[0]), "target_node" : int(working_node_match[1]), "score" : working_node_score,\
                                    "origin_node_attrs" : G1_nodes[working_node_match[0]], "target_node_attrs" : G2_nodes[working_node_match[1]]} for working_node_match in working_node_matches]
                    # self.logger.info(f"flag working_node_matches {working_node_matches}")
                else:
                    working_node_tuples = []
                    working_node_split_matches = working_node_attrs["split_match"]
                    # self.logger.info(f"flag working_node_split_matches {working_node_split_matches}")
                    working_node_split_scores = working_node_attrs["split_scores"]
                    for i, working_node_split_match in enumerate(working_node_split_matches):
                        for pair in working_node_split_match:
                            pair_dictionary = {"origin_node" : int(pair[0]), "target_node" : int(pair[1]), "score" : working_node_split_scores[i],\
                                    "origin_node_attrs" : G1_nodes[pair[0]], "target_node_attrs" : G2_nodes[pair[1]]}
                            working_node_tuples.append(pair_dictionary)

                ### OLD
                # working_node_matches = pruned_match_graph.get_attributes_of_node(working_node_ID)["match"]
                # working_node_score = pruned_match_graph.get_attributes_of_node(working_node_ID)["score_intralevel"]
                # working_node_tuples = [{"origin_node" : int(working_node_match[0]), "target_node" : int(working_node_match[1]), "score" : working_node_score,\
                #                 "origin_node_attrs" : G1_nodes[working_node_match[0]], "target_node_attrs" : G2_nodes[working_node_match[1]]} for working_node_match in working_node_matches]
                ### END
                
                if len(swept_levels) > lvl:
                    neighbour_nodes_IDs = pruned_match_graph.get_neighbourhood_graph(working_node_ID).find_nodes_by_attrs({"type": swept_levels[lvl]})
            else:
                working_node_tuples = []
                if len(swept_levels) > lvl:
                    neighbour_nodes_IDs = pruned_match_graph.find_nodes_by_attrs({"type": swept_levels[lvl]})

            stacked_tuples = []
            if len(swept_levels) > lvl:
                # self.logger.info(f"flag neighbour_nodes_IDs {neighbour_nodes_IDs}")
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

        def plot_symmetry_detection_scores(X, best_cluster_candidates):
            X_best = np.array([match["consistency_avg"] for match in best_cluster_candidates])
            x1 = [1] * len(X)
            x2 = [1] * len(X_best)

            # Plot each vector with different x positions
            fig = plt.figure("Symmetries detection")
            fig.clf()
            plt.plot(x1, X, 'o-', label='Candidates')
            plt.plot(x2, X_best, 'o-', label='Selected')
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Symmetries detection")
            # plt.show()

        # plot_symmetry_detection_scores(X, best_cluster_candidates)
        # time.sleep(555)

        return best_cluster_candidates


    def prune_interlevel(self,match_graph, G1_full, G2_full, merged_levels):

        higher_level_group_nodes = list(match_graph.filter_graph_by_node_types(merged_levels[0])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids())

        consistent_combinations = []
        for node in higher_level_group_nodes:
            new_consistent_combinations = self.merge_lower_level_groups(match_graph, G1_full, G2_full, node, merged_levels)
            if new_consistent_combinations:
                consistent_combinations += new_consistent_combinations

        # ### DEBUGGING
        # def make_hashable(value):
        #     """Recursively make the value hashable."""
        #     if isinstance(value, (list, set)):
        #         return tuple(make_hashable(v) for v in value)
        #     elif isinstance(value, dict):
        #         return frozenset((k, make_hashable(v)) for k, v in value.items())
        #     return value

        # self.logger.info(f"dbg len consistent_combinations {len(consistent_combinations)}")
        # self.stored_consistent_combinations.append(consistent_combinations)
        # self.logger.info(f"dbg self.stored_consistent_combinations {len(self.stored_consistent_combinations)}")
        # if len(self.stored_consistent_combinations) > 1:
        #     for i in range(len(self.stored_consistent_combinations) - 1):
        #         set1 = {make_hashable(d) for d in self.stored_consistent_combinations[i]}
        #         set2 = {make_hashable(d) for d in self.stored_consistent_combinations[i+1]}
        #         condition = set1 == set2
        #         self.logger.info(f"dbg condition {condition}")
        # ### END DEBUGGING
        self.select_high_level_groups(match_graph, consistent_combinations, merged_levels)


    def merge_lower_level_groups(self, match_graph, G1_full, G2_full, working_node_ID, merged_levels):

        higher_level_single_pairs_nodes = list(match_graph.get_neighbourhood_graph(working_node_ID)\
                                                            .filter_graph_by_node_types(merged_levels[0])\
                                                            .filter_graph_by_node_attributes({"combination_type" : "pair"})\
                                                            .get_nodes_ids())
        # self.logger.info(f"dbg len(higher_level_single_pairs_nodes) {higher_level_single_pairs_nodes}")
        lower_level_group_nodes = [list(match_graph.get_neighbourhood_graph(node)\
                                                    .filter_graph_by_node_types(merged_levels[1])\
                                                    .filter_graph_by_node_attributes({"combination_type" : "group"})\
                                                    .get_nodes_ids()) for node in higher_level_single_pairs_nodes]
        # self.logger.info(f"dbg len(lower_level_group_nodes) {lower_level_group_nodes}")

        combinations = multilist_combinations(lower_level_group_nodes)
        
        # self.logger.info(f"flag downstream scores {[match_graph.get_attributes_of_node(i).get('downstream_score') for i in higher_level_single_pairs_nodes]}")
        
        dbg_room_match = match_graph.get_attributes_of_node(working_node_ID).get('match')
        best_parent_index = np.argmax([match_graph.get_attributes_of_node(i).get('downstream_score') for i in higher_level_single_pairs_nodes])
        
        parent_node_attrs = match_graph.get_attributes_of_node(higher_level_single_pairs_nodes[best_parent_index])
        
        # parent1_data = self.change_pos_dt(G1_full, [parent_node_attrs["match"][0]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        # parent2_data = self.change_pos_dt(G2_full, [parent_node_attrs["match"][1]], self.params["levels"]["datatype"][merged_levels[0]], self.params["levels"]["datatype"][merged_levels[1]])
        consistent_combinations = []
        for combination in combinations:
            A_categorical = set()
            for node in combination:
                A_categorical.update(match_graph.get_attributes_of_node(node)["match"])
            data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, A_categorical, "Geometric_info")
            # data1, data2, A_numerical, floor_pair_numerical = self.add_floor_data(data1, data2, A_numerical)

            if self.stored_match_graph:
                stored_match_graph = copy.deepcopy(self.stored_match_graph)
                stored_match_graph_lvl_match = stored_match_graph.get_attributes_of_node(stored_match_graph.find_nodes_by_attrs({"type": merged_levels[1], "combination_type" : "group","merge_lvl": 1})[0])["match"]
                stored_match_graph_lvl_data = np.array([(copy.deepcopy(G1_full.get_attributes_of_node(match[0])["Geometric_info"]), copy.deepcopy(G2_full.get_attributes_of_node(match[1])["Geometric_info"])) for match in stored_match_graph_lvl_match])
                data1, data2, A_numerical = self.add_parents_data(data1, data2, A_numerical, stored_match_graph_lvl_data[:,0,:], stored_match_graph_lvl_data[:,1,:])
                # data1 = self.geometric_info_transformation(data1, merged_levels[1], G1_full.get_attributes_of_node(parent_node_attrs["match"][0])["Geometric_info"])
                # data2 = self.geometric_info_transformation(data2, merged_levels[1], G2_full.get_attributes_of_node(parent_node_attrs["match"][1])["Geometric_info"])
                # clipper = Clipper(self.params["levels"]["datatype"][merged_levels[1]], self.params["levels"]["clipper_invariants"][merged_levels[1]], self.params, self.logger)
                # clipper.score_pairwise_consistency(data1, data2, A_numerical)
                # consistency_avg = clipper.get_score_all_inital_u()
            
            data1 = self.geometric_info_transformation(data1, merged_levels[1], G1_full.get_attributes_of_node(parent_node_attrs["match"][0])["Geometric_info"])
            data2 = self.geometric_info_transformation(data2, merged_levels[1], G2_full.get_attributes_of_node(parent_node_attrs["match"][1])["Geometric_info"])
            # self.logger.info(f"dbg merge A_categorical {A_categorical}")
            clipper = Clipper(self.params["levels"]["datatype"][merged_levels[1]], self.params["levels"]["clipper_invariants"][merged_levels[1]], self.params, self.logger)
            clipper.score_pairwise_consistency(data1, data2, A_numerical)
            consistency_avg = clipper.get_score_all_inital_u()
            # self.logger.info(f"dbg consistency_avg {consistency_avg}")
            # floor_condition = self.assess_floor_consistency(data1, data2, merged_levels[1])
            floor_condition = True

            if consistency_avg >= self.params["thresholds"]["global"] and floor_condition:
                # self.logger.info(f"dbg consistency_avg IN {consistency_avg}")
                consistent_combinations.append({"consistency_avg":consistency_avg,"lower_level_nodes_IDs": combination,"match":A_categorical, "higher_level_node_ID":working_node_ID})
            # for consistent_combination in consistent_combinations:
            #     self.logger.info(f"flag consistent_combination 1 {consistent_combination['match']}")

        return consistent_combinations


    def select_high_level_groups(self, match_graph, consistent_combinations, merged_levels):
        if consistent_combinations:
            # for consistent_combination in consistent_combinations:
            #     self.logger.info(f"flag consistent_combination 2 {consistent_combination['match']}")
            best_combinations = self.symmetry_detection(consistent_combinations)
            # self.logger.info(f"flag best_combinations {len(best_combinations)}")
            for best_combination in best_combinations:
                split_match, split_score = [], []

                for lower_level_node_id in best_combination["lower_level_nodes_IDs"]:
                    lower_level_node_attrs = match_graph.get_attributes_of_node(lower_level_node_id)
                    split_match.append(lower_level_node_attrs["match"])
                    split_score.append(lower_level_node_attrs["score_intralevel"])

                best_combination_node_id = match_graph.get_total_number_nodes() + 1
                node_attr = [(best_combination_node_id, {"type": merged_levels[1], "match": best_combination["match"], "merge_lvl" :1,\
                                            "combination_type" : "group", "score_intralevel" : best_combination["consistency_avg"],
                                            "split_match" : split_match, "split_scores" : split_score, "best_pair" : False})]
                edges_attr = [(lower_level_node, best_combination_node_id, {}) for lower_level_node in best_combination["lower_level_nodes_IDs"]]
                edges_attr.append((best_combination["higher_level_node_ID"], best_combination_node_id, {}))
                match_graph.add_subgraph(node_attr, edges_attr)

                match_graph.set_node_attributes("merge_lvl", {best_combination["higher_level_node_ID"]:1})


    def select_best_global_localization_pair(self, match_graph, swept_levels):
        MG_ws_nodes = match_graph.find_nodes_by_attrs({"type": swept_levels[-1], "combination_type" : "group","merge_lvl": len(swept_levels)-1})
        # self.logger.info("flag bestPair MG_ws_nodes {}".format(MG_ws_nodes))

        for ws_node in MG_ws_nodes:
            attrs = match_graph.get_attributes_of_node(ws_node)
            # self.logger.info("flag bestPair attrs['split_match'] {}".format(attrs['split_match']))
            split_matches_lengths = [len(split_match) for split_match in attrs['split_match']]
            # self.logger.info("flag bestPair split_matches_lengths {}".format(split_matches_lengths))
            longest_split_matches_idx = np.array(split_matches_lengths) == max(split_matches_lengths)
            # self.logger.info("flag bestPair longest_split_matches_idx {}".format(longest_split_matches_idx))
            # self.logger.info("flag bestPair attrs[split_scores] {}".format(attrs['split_scores']))
            best_score_index = np.array(attrs["split_scores"])[longest_split_matches_idx].argmax()
            # self.logger.info("flag bestPair best_score_index {}".format(best_score_index))
            # self.logger.info("flag bestPair 1 {}".format(attrs["split_match"]))
            # self.logger.info("flag bestPair 2 {}".format(np.array(attrs["split_match"])[longest_split_matches_idx]))
            # self.logger.info("flag bestPair 3 {}".format(np.array(attrs["split_match"])[longest_split_matches_idx][best_score_index]))
            best_score_pair = list(np.array(attrs["split_match"])[longest_split_matches_idx][best_score_index])[0]
            match_graph.set_node_attributes("best_pair", {ws_node : best_score_pair})
            # self.logger.info("flag bestPair attrs[split_scores] {}".format(attrs['split_scores']))
            # self.logger.info("flag bestPair attrs['split_match'] {}".format(attrs['split_match']))
            # self.logger.info("flag bestPair best_score_index {}".format(best_score_index))
            # self.logger.info("flag bestPair best_score_pair {}".format(best_score_pair))

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
                parent1_data = copy.deepcopy(G1_full.get_attributes_of_node(G1_matched_nodes[0])["Geometric_info"])
                parent2_data = copy.deepcopy(G2_full.get_attributes_of_node(G2_matched_nodes[0])["Geometric_info"])
                data1, data2, all_pairs_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, wild_nodes_combination, "Geometric_info")
                clipper = Clipper(self.params["levels"]["datatype"][swept_levels[1]], self.params["levels"]["clipper_invariants"][swept_levels[1]], self.params, self.logger)
                data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(data1, data2, all_pairs_numerical, parent1_data, parent2_data)
                data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[1], parent1_data))
                data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[1], parent2_data))
                clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                M_aux, _ = clipper.get_M_C_matrices()
                interlevel_scores = M_aux[:,-1][:-1]
                good_pairs = interlevel_scores >= self.params["thresholds"]["local_interlevel"][f"{swept_levels[0]} - {swept_levels[1]}"][0]
                bad_pairs = [not elem for elem in good_pairs]
                filtered_bad_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[bad_pairs], nodes1, nodes2))
                filtered_good_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_numerical[good_pairs], nodes1, nodes2))
                # interlevel_scores_dict = {list(filtered_good_pairs_categorical)[i]: interlevel_scores[good_pairs][i] for i in range(len(filtered_good_pairs_categorical))}

                merged_node_match.update(filtered_good_pairs_categorical)

                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1_full, G2_full, merged_node_match, "Geometric_info")
                data1 = self.geometric_info_transformation(data1, swept_levels[1], parent1_data)
                data2 = self.geometric_info_transformation(data2, swept_levels[1], parent2_data)
                # data1, data2, A_numerical, floor_pair_numerical = self.add_floor_data(data1, data2, A_numerical)
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

    def add_deviated_nodes_by_level(self, MG_full, G1_full, G2_full, swept_levels):
        # Best Pair data
        MG_room_nodes = MG_full.find_nodes_by_attrs({"type": swept_levels[0], "combination_type" : "group","merge_lvl": 1})
        MG_planes_nodes = MG_full.find_nodes_by_attrs({"type": swept_levels[-1], "combination_type" : "group","merge_lvl": 1})
        
        if len(MG_room_nodes) == 1 and len(MG_planes_nodes) == 1:
            MG_ws_node_attrs = MG_full.get_attributes_of_node(MG_planes_nodes[0])
            MG_ws_node_attrs_dev = self.stored_match_graph_dev.get_attributes_of_node(0)

            MG_planes_match = MG_ws_node_attrs["match"]
            # self.logger.info(f"flag uniquely matched plane pairs {len(MG_planes_match), MG_planes_match}")
            MG1_planes_match_nodes = np.array(list(MG_planes_match))[:,0]
            MG2_planes_match_nodes = np.array(list(MG_planes_match))[:,1]
            all_walls_gi1, all_walls_gi2, all_walls_A_numerical, all_walls_nodes1, all_walls_nodes2 = self.generate_clipper_input(G1_full, G2_full, MG_ws_node_attrs["match"], "Geometric_info")
            best_pair = MG_ws_node_attrs["best_pair"]
            best_pair_data1 = copy.deepcopy(G1_full.get_attributes_of_node(best_pair[0])["Geometric_info"])
            best_pair_data2 = copy.deepcopy(G2_full.get_attributes_of_node(best_pair[1])["Geometric_info"])
            self.best_pair_data = [best_pair_data1, best_pair_data2]
            MG_rooms_match = MG_full.get_attributes_of_node(MG_room_nodes[0])["match"]

            for MG_room_match in MG_rooms_match:
                G1_room_planes = G1_full.get_neighbourhood_graph(MG_room_match[0]).filter_graph_by_node_attributes({"type": swept_levels[1]}).get_nodes_ids()
                # self.logger.info(f"flag G1_room_planes {len(G1_room_planes), G1_room_planes}")
                G2_room_planes = G2_full.get_neighbourhood_graph(MG_room_match[1]).filter_graph_by_node_attributes({"type": swept_levels[1]}).get_nodes_ids()
                # self.logger.info(f"flag G2_room_planes {len(G2_room_planes), G2_room_planes}")
                G1_unmatched_planes = list(set(G1_room_planes) - set(MG1_planes_match_nodes))
                # self.logger.info(f"flag G1_unmatched_planes {len(G1_unmatched_planes), G1_unmatched_planes}")
                G2_unmatched_planes = list(set(G2_room_planes) - set(MG2_planes_match_nodes))
                # self.logger.info(f"flag G2_unmatched_planes {len(G2_unmatched_planes), G2_unmatched_planes}")

                if len(G1_unmatched_planes) != 0 and len(G2_unmatched_planes) != 0:
                    ### Use clipper utility function to compute consistency
                    wild_nodes_combination = multilist_combinations([G1_unmatched_planes, G2_unmatched_planes])
                    wild_nodes_data1, wild_nodes_data2, wild_pairs_numerical, wild_nodes1, wild_odes2 = self.generate_clipper_input(G1_full, G2_full, wild_nodes_combination, "Geometric_info")
                    data1, data2, all_pairs_and_parent_numerical = self.add_parents_data(wild_nodes_data1, wild_nodes_data2, wild_pairs_numerical, all_walls_gi1, all_walls_gi2)
                    data1 = copy.deepcopy(self.geometric_info_transformation(data1, swept_levels[1], self.best_pair_data[0]))
                    data2 = copy.deepcopy(self.geometric_info_transformation(data2, swept_levels[1], self.best_pair_data[1]))
                    clipper = Clipper(self.params["levels"]["datatype"][swept_levels[1]], "1", self.params, self.logger)
                    clipper.score_pairwise_consistency(data1, data2, all_pairs_and_parent_numerical)
                    M_aux, _ = clipper.get_M_C_matrices()
                    entry_interlevel_scores = M_aux[len(wild_nodes_combination):,:len(wild_nodes_combination)]
                    # self.logger.info(f"flag entry_interlevel_scores {entry_interlevel_scores}")
                    interlevel_scores = np.sum(entry_interlevel_scores, axis = 0) / (len(all_pairs_and_parent_numerical) - len(wild_nodes_combination))
                    # self.logger.info(f"dbg deviations interlevel_scores {interlevel_scores}")
                    good_pairs = interlevel_scores >= self.params["thresholds"]["global_intralevel_deviations"][f"{swept_levels[1]}"]
                    filtered_good_pairs_score = interlevel_scores[good_pairs]
                    filtered_good_pairs_categorical = set(clipper.categorize_clipper_output(all_pairs_and_parent_numerical[:len(wild_nodes_combination)][good_pairs], wild_nodes1, wild_odes2))
                    # self.logger.info(f"flag good / all pairs {len(filtered_good_pairs_categorical)} {len(interlevel_scores)}")

                    for i, filtered_good_pair_categorical in enumerate(filtered_good_pairs_categorical):
                        # self.logger.info(f"dbg including DEVIATION of pair {filtered_good_pair_categorical}")
                        MG_ws_node_attrs["match"].update(set([filtered_good_pair_categorical]))
                        MG_ws_node_attrs["split_match"].append(set([filtered_good_pair_categorical]))
                        MG_ws_node_attrs["split_scores"].append(filtered_good_pairs_score[i])
                        # self.logger.info(f"flag MG_ws_node_attrs {MG_ws_node_attrs}")
                        MG_ws_node_attrs_dev["match"].update(set([filtered_good_pair_categorical]))
                        MG_ws_node_attrs_dev["split_match"].append(set([filtered_good_pair_categorical]))
                        MG_ws_node_attrs_dev["split_scores"].append(filtered_good_pairs_score[i])
                        MG_ws_node_attrs_dev["score_intralevel"] = filtered_good_pairs_score[i]

    def draw_as_match_graph(self, match_graph, name):
        node_color = match_graph.define_draw_color_option_by_node_type()
        node_size = match_graph.define_node_size_option_by_combination_type_attr()
        linewidths = match_graph.define_node_linewidth_option_by_combination_type_attr()
        options = {'node_color': node_color, 'node_size': 50, 'width': 2, 'with_labels' : True,\
                "node_size" : node_size, "linewidths" : linewidths, "edgecolors" : "black"}
        match_graph.draw(name, options = options, show = self.params["verbose"])


    def plot_geometry_set(self, title, datatype, data, ax = None, color = "red", tags = None):  
        if not ax:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        points = np.array(data)[:, :3]
        if datatype == "points&normal":
            normals = np.array(data)[:,3:]
        
        if datatype == "points":
            # Plot the points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label='Rooms', s = 50, marker = "s")

        # Plot the normals
        elif datatype == "points&normal":
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label='Planes')
            for point, normal in zip(points, normals):
                ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.3, arrow_length_ratio=0.1, color='red')
        
        # Plot point labels
        for i in range(len(points)):
            if tags is not None:
                tag = str(tags[i])
            else:
                tag = str(i)
            ax.text(points[i, 0], points[i, 1], points[i, 2], tag, color = "black")
            
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()

        return points

    def plot_geometry_graphs(self, graphs, swept_levels):
        fig = plt.figure("plot_geometry_graphs", figsize=(20, 14))
        all_points = np.empty((0, 3))
        axs = []
        colors = ["blue", "green"]
        for i_graph, graph in enumerate(graphs):
            plot_number = 100 + 10 * len(graphs) + i_graph + 1
            ax = fig.add_subplot(plot_number, projection='3d')
            axs.append(ax)
            for swept_level in swept_levels:
                attrs_all_lvl_nodes = graph.filter_graph_by_node_types(swept_level).get_attributes_of_all_nodes()
                data = [attrs[1]["Geometric_info"] for attrs in attrs_all_lvl_nodes]
                node_ids = [pair[0] for pair in attrs_all_lvl_nodes]
                points = self.plot_geometry_set(graph.name, self.params["levels"]["datatype"][swept_level], data, ax, colors[i_graph], node_ids)
                all_points = np.vstack((all_points, points))

        # all_points = np.vstack((points1, points2))
        x_limits = (all_points[:, 0].min(), all_points[:, 0].max())
        y_limits = (all_points[:, 1].min(), all_points[:, 1].max())
        z_limits = (1, -1)

        for ax in axs:
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)

        plt.show()

    def plot_geometry_setlist(self, figure_name, set_list, datatype, tags=None):
        fig = plt.figure(figure_name, figsize=(20, 14))
        all_points = np.empty((0, 3))
        axs = []
        colors = ["blue", "green"]
        for i_set_list, data in enumerate(set_list):
            # print(f"dbg plotting set {i_set_list} with {len(data)} elements")
            # print(f"dbg data {data}")
            plot_number = 100 + 10 * len(set_list) + i_set_list + 1
            ax = fig.add_subplot(plot_number, projection='3d')
            axs.append(ax)
            # tags = A_numerical[:,i_set_list] if A_numerical is not None else None
            if tags:
                points = self.plot_geometry_set(str(i_set_list), datatype, data, ax, colors[i_set_list], tags[i_set_list])
            else:
                points = self.plot_geometry_set(str(i_set_list), datatype, data, ax, colors[i_set_list])
            
            all_points = np.vstack((all_points, points))

        # all_points = np.vstack((points1, points2))
        x_limits = (all_points[:, 0].min(), all_points[:, 0].max())
        y_limits = (all_points[:, 1].min(), all_points[:, 1].max())
        z_limits = (1, -1)

        xy_limits = (min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1]))

        for ax in axs:
            ax.set_xlim(xy_limits)
            ax.set_ylim(xy_limits)
            ax.set_zlim(z_limits)

        plt.show(block=False)
        plt.pause(0.1)

    def plot_comparison_geometry_setlist(self, axs, set_list, datatype, tags=None):
        titles = ["Prior", "Online"]
        all_points = np.empty((0, 3))
        colors = ["blue", "green"]
    
        for i_set_list, data in enumerate(set_list):
            ax = axs[i_set_list]
    
            points = self.plot_geometry_set(
                titles[i_set_list],
                datatype,
                data,
                ax,
                colors[i_set_list],
                tags
            )
            all_points = np.vstack((all_points, points))
    
        # shared limits inside the pair
        x_limits = (all_points[:, 0].min(), all_points[:, 0].max())
        y_limits = (all_points[:, 1].min(), all_points[:, 1].max())
        xy_limits = (min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1]))
        z_limits = (1, -1)
    
        for ax in axs:
            ax.set_xlim(xy_limits)
            ax.set_ylim(xy_limits)
            ax.set_zlim(z_limits)
        ax.legend_.remove()

    def generate_comparison_plots(self, outer_gs, fig, pair_i, data1, data2, A_categorical, swept_levels, lvl, tags):

        inner_gs = outer_gs[0, pair_i].subgridspec(2, 1, hspace=0.4)

        axs = []
        for j in range(2):
            ax = fig.add_subplot(inner_gs[j, 0], projection="3d")
            axs.append(ax)

        self.plot_comparison_geometry_setlist(axs, [data1, data2], self.params["levels"]["datatype"][swept_levels[lvl]], tags)

        bbox = outer_gs[0, pair_i].get_position(fig)
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_top = bbox.y1
        y_bot = bbox.y0
        title_subplot_h = y_bot - 0.01
        if (pair_i % 2 == 0):
            title_subplot_h = y_top + 0.01
        title = f"{list(A_categorical)}"
        print("TITLE: ", title)
        fig.text(
            x_center,
            title_subplot_h,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        fig.canvas.draw_idle()
        plt.pause(0.1)

    def get_room_content(self, full_graph, room_node):
        """
        This function returns the content of a room node in the full graph.
        It retrieves the attributes of all nodes in the room and returns them as a list.
        """
        room_content_nodes = {}
        room_content_nodes[self.ws_string] = []
        room_neigbourhood = list(full_graph.get_neighbourhood_graph(room_node).get_nodes_ids())
        for node in room_neigbourhood:
            if full_graph.get_attributes_of_node(node).get("type") == self.ws_string:
                room_content_nodes[self.ws_string].append(node)
                ws_neighbourhood = list(full_graph.get_neighbourhood_graph(node).get_nodes_ids())
                for node_id in ws_neighbourhood:
                    node_type = full_graph.get_attributes_of_node(node_id).get("type")
                    if node_type in ["wall", self.ws_string, self.room_string]:
                        continue
                    if node_type not in room_content_nodes.keys():
                        room_content_nodes[node_type] = []
                    if node_id not in room_content_nodes[node_type]:
                        room_content_nodes[node_type].append(node_id)

        return room_content_nodes

    def get_ws_content(self, full_graph, ws_node):
        """
        This function returns the content of a workspace node in the full graph.
        It retrieves the attributes of all nodes in the workspace and returns them as a list.
        """
        ws_content_nodes = {}
        # ws_content_nodes["ws"] = [ws_node]
        ws_neigbourhood = list(full_graph.get_neighbourhood_graph(ws_node).get_nodes_ids())
        for node in ws_neigbourhood:
            node_type = full_graph.get_attributes_of_node(node).get("type")
            if node_type in ["wall", self.ws_string, self.room_string]:
                continue
            if node_type not in ws_content_nodes.keys():
                ws_content_nodes[node_type] = []
            if node not in ws_content_nodes[node_type]:
                ws_content_nodes[node_type].append(node)

        return ws_content_nodes

    def filter_by_content(self, all_pairs_categorical, G1_full, G2_full, G1_lvl, G2_lvl):
        # print("FILTER BY CONTENT started")
        # get rooms content
        content = {}
        for node in G1_lvl.get_nodes_ids():
            node_type = G1_full.get_attributes_of_node(node).get("type")
            # print(f"flag node {node} type {node_type}")
            if node_type == self.room_string:
                content[node] = self.get_room_content(
                        G1_full, node)
            if node_type == self.ws_string:
                content[node] = self.get_ws_content(
                        G1_full, node)
        for node in G2_lvl.get_nodes_ids():
            node_type = G2_full.get_attributes_of_node(node).get("type")
            if node_type == self.room_string:
                content[node] = self.get_room_content(
                        G2_full, node)
            if node_type == self.ws_string:
                content[node] = self.get_ws_content(
                        G2_full, node)
        
        remove_pairs = []
        for pair in all_pairs_categorical:
            # print(f"fbc checking pair {pair}")
            # print(f"fbc content {content[pair[1]]} vs {content[pair[0]]}")

            if content[pair[0]].keys() != content[pair[1]].keys():
                # print(f"fbc removing pair {pair} because of different keys")
                remove_pairs.append(pair)
                continue

            for key in content[pair[1]].keys():
                # print(f"fbc checking key {key} from {pair[1]} to {pair[0]}")
                if key not in content[pair[0]].keys():
                    # print(f"fbc removing pair {pair} because of key {key}")
                    remove_pairs.append(pair)
                    # all_pairs_categorical.remove(pair)
                    # for ws_pair in all_pairs_categorical:
                    #     print(f"flag ws_pair {ws_pair} in pair {pair}")
                    #     if ws_pair[0] in content[pair[0]][self.ws_string] and ws_pair[1] in content[pair[0]][self.ws_string]:
                    #         print(f"fbc removing ws pair {ws_pair} because of key {key}")
                    #         all_pairs_categorical.remove(ws_pair)
                    break

                # If same keys, check if same number of object_same_room
                # print(f"fbc comparing lengths {len(content[pair[0]][key])} vs {len(content[pair[1]][key])}")
                if len(content[pair[0]][key]) != len(content[pair[1]][key]):
                    # print(f"flag removing pair {pair} because of different number of {key}")
                    # all_pairs_categorical.remove(pair)
                    remove_pairs.append(pair)
                    break
        # print("FILTER BY CONTENT finished")

        return remove_pairs

