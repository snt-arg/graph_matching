import numpy as np
from GraphManager import GraphManager
from Clipper import Clipper

SCORE_THR = 0.9


class GraphMatcher():
    def __init__(self):
        self.graphs = {}
        self.graphs["match"] = GraphManager(graph_def={'name': "match",'nodes' : [], 'edges' : []})


    def setGraph(self, graph_def):
        self.graphs[graph_def['name']] = GraphManager(graph_def = graph_def)


    def new_match_custom(self, G1_name, G2_name):
        sweeped_levels = ["floor", "room", "wall"]
        sweeped_levels_dt = ["points", "points", "points&normal"]
        full_graph_matches = self.graphs[G1_name].matchByNodeType(self.graphs[G2_name])
        lvl = 0

        def match_iteration(G1, G2, lvl, parents_data = None):
            G1_lvl = G1.filter_graph_by_node_types(sweeped_levels[lvl])
            G2_lvl = G2.filter_graph_by_node_types(sweeped_levels[lvl])
            matches = G1_lvl.matchByNodeType(G2_lvl)
            # A_categorical, _ = self.matches_to_suitable_A_C(G1_lvl, matches, full_graph_matches, [sweeped_levels[lvl]])
            matches = self.filter_local_match_with_global(matches, full_graph_matches)
            scores = []
            submatches = []
            nodes_id = []
            for A_categorical in matches:
                data1, data2, A_numerical, nodes1, nodes2 = self.generate_clipper_input(G1, G2, A_categorical, "pos")
                if parents_data:
                    # print("flag")
                    # print(data1, data2, A_numerical)
                    self.add_parents_data(data1, data2, A_numerical, parents_data)
                    # print(data1, data2, A_numerical)
                clipper = Clipper(sweeped_levels_dt[lvl])
                clipper.score_pairwise_consistency(data1, data2, A_numerical)
                if parents_data:
                    M_aux, _ = clipper.get_M_C_matrices()
                    clipper.remove_last_assoc_M_C_matrices()
                    # clipper.filter_C_M_matrices(C_graph) # TODO: waiting for an answer in the issue
                    # clipper.set_M_diagonal_values(M_aux[-1,:][:-1]) # TODO: waiting for an answer in the issue
                clipper_match_numerical, score = clipper.solve_clipper()
                clipper_match_categorical = clipper.categorize_clipper_output(clipper_match_numerical, nodes1, nodes2)
                scores.append(score)
                submatches.append(clipper_match_categorical)
                
                print("Found candidates with score {} for matching: \n {}".format(score, clipper_match_categorical))
                if score > SCORE_THR:
                    node_id = self.graphs["match"].get_total_number_nodes() + 1
                    nodes_id.append(node_id)
                    print(node_id)
                    node_attr = [(node_id, {"type": sweeped_levels[lvl], "match": A_categorical})]
                    if parents_data:
                        edges_attr = [(node_id, parents_data["id"])]
                    else:
                        edges_attr = []
                    self.graphs["match"].add_subgraph(node_attr, edges_attr)

            # sorted_matches_indexes = np.argsort(scores)[::-1]
            # sorted_scores = np.sort(scores)[::-1]
            best_matches_indeces = np.where(np.array(scores) > SCORE_THR)[0]
            best_scores = [scores[i] for i in best_matches_indeces]
            best_submatches = [submatches[i] for i in best_matches_indeces]
            print(best_scores)
            print(best_submatches)


            ## Next level
            if lvl < len(sweeped_levels) - 1:
                print("best_submatches", best_submatches)
                for good_submatch in best_submatches:
                    print("good_submatch", good_submatch)
                    parent1_data = self.change_pos_dt(G1, good_submatch[:,0], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
                    parent2_data = self.change_pos_dt(G2, good_submatch[:,1], sweeped_levels_dt[lvl], sweeped_levels_dt[lvl+1])
                    scores = []
                    subsubmatches = []
                    for i, lowerlevel_match in enumerate(good_submatch):
                        print("Checking lowerlevel matches for {}\n".format(lowerlevel_match))
                        G1_neighborhood = self.graphs[G1_name].get_neighbourhood_graph(lowerlevel_match[0])
                        G2_neighborhood = self.graphs[G2_name].get_neighbourhood_graph(lowerlevel_match[1])
                        parents_data = {"match" : lowerlevel_match, "parent1" : parent1_data[i], "parent2" : parent2_data[i], "id" : nodes_id[i]}
                        subsubmatch, score = match_iteration(G1_neighborhood, G2_neighborhood, lvl + 1, parents_data)
                        scores.append(score)
                        subsubmatches.append(subsubmatch)

                
            return(best_submatches, best_scores)


        match_iteration(self.graphs[G1_name], self.graphs[G2_name], lvl)
        self.graphs["match"].draw("match", options = None, show = True)




    def filter_local_match_with_global(self, local_matches, global_matches):
        filtered = [local_match for local_match in local_matches if any(all(elem in global_match for elem in local_match) for global_match in global_matches)]
        filtered_as_arrays = [np.array(x, dtype = np.str) for x in filtered]
        return filtered_as_arrays
        

    def generate_clipper_input(self, G1, G2, A_categorical, feature_name):
        nodes1, nodes2 = list(set(A_categorical[:,0])), list(set(A_categorical[:,1]))
        data1 = G1.stack_nodes_feature(nodes1, feature_name)
        data2 = G2.stack_nodes_feature(nodes2, feature_name)
        A_numerical = np.array([[nodes1.index(pair[0]),nodes2.index(pair[1])] for pair in A_categorical]).astype(np.int32)
        return(data1, data2, A_numerical, nodes1, nodes2)


    def change_pos_dt(self, graph, match, in_dt, out_dt):
        original = graph.stack_nodes_feature(match, "pos")

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