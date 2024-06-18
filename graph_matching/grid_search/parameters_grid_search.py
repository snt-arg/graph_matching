import numpy as np
import graph_matching
from graph_matching.GraphMatcher import GraphMatcher
from graph_matching.utils import plane_4_params_to_6_params, compute_room_center
import os, json
import pandas as pd
import itertools
import copy, random
import time
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import timeout_decorator

from graph_wrapper.GraphWrapper import GraphWrapper

class PrintLogger():
    def __init__(self):
        pass
    def info(self, msg):
        print(msg)

def parse_room_file(file_path):
    # Step 1: Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Step 2: Initialize variables to store the parsed data
    ids = {}
    vectors = {}
    matrices = {}
    current_key = None
    collect_matrix = False
    matrix_data = []

    # Step 3: Parse the data
    for line_i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit():
            current_key = None
        elif line:
            if 'id' in line and line != "room_keyframes_ids":
                key = line.strip()
                value = int(lines[line_i + 1].strip())
                ids[key] = value
                current_key = None
            elif 'room_node' in line:
                current_key = 'room_node'
                collect_matrix = True
            elif 'Plane' in line or 'node' in line:
                current_key = line.strip()
                vectors[current_key] = []
            elif current_key and collect_matrix:
                if line:
                    matrix_data.append(list(map(float, line.split())))
                    if len(matrix_data) == 4:  # Assuming 4x4 matrix
                        matrices[current_key] = np.array(matrix_data)
                        matrix_data = []
                        collect_matrix = False
                        current_key = None
            elif current_key:
                vectors[current_key].append(float(line))
                if len(vectors[current_key]) == 4:
                    vectors[current_key] = np.array(vectors[current_key])
                    current_key = None

    # # Convert remaining vectors to numpy arrays if not done already
    # for key in vectors:
    #     if isinstance(vectors[key], list):
    #         vectors[key] = np.array(vectors[key])

    # # Step 4: Print or use the parsed data
    # print("IDs:", ids)
    # print("Vectors:")
    # for key, vector in vectors.items():
    #     print(f"{key}: {vector}")

    # print("Matrices:")
    # for key, matrix in matrices.items():
    #     print(f"{key}:\n{matrix}")

    nodes = []
    geometric_info = matrices[f"room_node"][:3,-1]
    nodes.append((str(ids["id"]), {"type": "Finite Room", "Geometric_info": geometric_info, "draw_pos": geometric_info[:2]}))

    for plane_tag in ["x1", "x2", "y1", "y2"]:
        geometric_info = plane_4_params_to_6_params(vectors[f"plane_{plane_tag}_node"])
        nodes.append((str(ids[f"plane_{plane_tag}_id"]), {"type": "Plane", "Geometric_info": geometric_info, "draw_pos": geometric_info[:2]}))
    
    edges = []
    for plane_tag in ["x1", "x2", "y1", "y2"]:
        edges.append((str(ids["id"]), str(ids[f"plane_{plane_tag}_id"])))
    graph_dict = {"nodes": nodes, "edges": edges}
    
    return graph_dict

def parse_full_graph(name, folder_path):
    if os.path.exists(folder_path):
        folders = os.listdir(folder_path)
        graph_dict = {"name": name, "nodes": [], "edges": []}
        for folder_name in folders:
            if folder_name[0] == "0" and "room_data" in os.listdir(f"{folder_path}/{folder_name}"):
                single_room_dict = parse_room_file(f"{folder_path}/{folder_name}/room_data")
                graph_dict["nodes"] += single_room_dict["nodes"]
                graph_dict["edges"] += single_room_dict["edges"]
    else:
        print(f"WARNING: {folder_path} does not exist")

    return graph_dict

def score_estimated_match(estimated_match, gt_match):
    FP_penalization = 0
    FN_penalization = 1
    if len(estimated_match) == 1:
        gt_match_pairs = set([tuple(pair) for pair in gt_match['not_deviated']['Finite Room']] + [tuple(pair) for pair in gt_match['not_deviated']['Plane']])
        estimated_match_pairs = set([(pair['origin_node'], pair['target_node']) for pair in estimated_match[0]])
        true_positives = len([t for t in estimated_match_pairs if t in gt_match_pairs])
        false_positives = len(estimated_match_pairs) - true_positives
        false_negatives = len(gt_match_pairs) - true_positives
        score = (true_positives - FP_penalization*false_positives - FN_penalization*false_negatives)/len(gt_match_pairs)

    elif len(estimated_match) == 1:
        score = 0.0

    elif len(estimated_match) > 1:
        score = None
    
    return score

def one_experiment(deviated_graphs_dict, matching_params):
    gm = GraphMatcher(PrintLogger(), log_level=0)
    gm.set_parameters(matching_params)
    gm.set_graph_from_wrapper(deviated_graphs_dict["Prior"],"Prior")
    gm.set_graph_from_wrapper(deviated_graphs_dict["Online"],"Online")
    gt_matches = deviated_graphs_dict["GT"]

    success, matches, matches_full, matches_dev = gm.match("Prior", "Online")
    # print(f"dbg matches {len(matches)}")
    # if len(matches) > 0:
    #     print(f"dbg matches {len(matches[0])}")
    # print(f"dbg matches dev {len(matches_dev)} {matches_dev}")
    if matches:
        # for i, match in enumerate(matches):
        #     print(F"dbg match {i}")
        #     for pair in match:
        #         print(f"dbg [{pair['origin_node']} , {pair['target_node']}] {pair['origin_node_attrs']['type']}")
        score = score_estimated_match(matches, gt_matches)
    else:
        score = 0.0
    # time.sleep(200)
    
    return score

def match_params_update(matching_params_comb, base_matching_params,param_space):

    def update_nested_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    matching_params = copy.deepcopy(base_matching_params)
    for i, param in enumerate(param_space):
        mapping = paramter_mapping[param.name]
        update_nested_dict(matching_params, mapping, matching_params_comb[i])

    return matching_params

def divide_shared_planes(agraph, sgraph, GT):
    ### A GRAPH
    agraph_rooms_ids = copy.deepcopy(agraph.filter_graph_by_node_types(["Finite Room"]).get_nodes_ids())
    rooms_in_gt_agraph = [str(a) for a in np.array(GT["Finite Room"])[:,0]]
    rooms_filter_agraph = [room for room in agraph_rooms_ids if room not in rooms_in_gt_agraph]
    agraph.remove_nodes(rooms_filter_agraph)
    agraph.filterout_unparented_nodes()

    agraph_planes_ids = copy.deepcopy(agraph.filter_graph_by_node_types(["Plane"]).get_nodes_ids())

    a_graph_mapping = {}

    for node_id in agraph_planes_ids:
        plane_connections = list(agraph.edges_of_node(node_id))
        for plane_connection in plane_connections[:-1]:
            max_agraph_node_id = max([int(j) for j in agraph.get_nodes_ids()])
            new_node_id = str(max_agraph_node_id + 1)
            a_graph_mapping.update({node_id: new_node_id})

            agraph_room_id = plane_connection[1]
            sgraph_gt_room_id = np.array(GT["Finite Room"])[np.array(GT["Finite Room"])[:,0] == int(agraph_room_id)][0][1]
            sgraph_plane_candidates = list(sgraph.get_neighbourhood_graph(str(sgraph_gt_room_id)).filter_graph_by_node_types(["Plane"]).get_nodes_ids())
            sgraph_gt_planes = np.array(GT["Plane"])[np.array(GT["Plane"])[:,0] == int(node_id)][:,1]
            sgraph_gt_planes = [str(k) for k in sgraph_gt_planes]
            sgraph_gt_plane_id = list(set(sgraph_gt_planes).intersection(set(sgraph_plane_candidates)))[0]
            
            agraph.remove_edges([(str(node_id), str(agraph_room_id))])
            agraph.add_nodes([(str(new_node_id), agraph.get_attributes_of_node(node_id))])
            agraph.add_edges([(str(new_node_id), str(agraph_room_id), {})])

            GT["Plane"].remove([int(node_id), int(sgraph_gt_plane_id)])
            GT["Plane"] = GT["Plane"] + [[int(new_node_id), int(sgraph_gt_plane_id)]]


    ### S GRAPH
    sgraph_planes_ids = copy.deepcopy(sgraph.filter_graph_by_node_types(["Plane"]).get_nodes_ids())
    s_graph_mapping = {}

    for node_id in sgraph_planes_ids:
        plane_connections = list(sgraph.edges_of_node(node_id))
        
        for plane_connection in plane_connections[:-1]:
            max_sgraph_node_id = max([int(i) for i in sgraph.get_nodes_ids()])
            new_node_id = str(max_sgraph_node_id + 1)
            s_graph_mapping.update({node_id: new_node_id})

            sgraph_room_id = plane_connection[1]
            agraph_gt_room_id = np.array(GT["Finite Room"])[np.array(GT["Finite Room"])[:,1] == int(sgraph_room_id)][0][0]
            agraph_plane_candidates = list(agraph.get_neighbourhood_graph(str(agraph_gt_room_id)).filter_graph_by_node_types(["Plane"]).get_nodes_ids())
            agraph_gt_planes = np.array(GT["Plane"])[np.array(GT["Plane"])[:,1] == int(node_id)][:,0]
            agraph_gt_planes = [str(k) for k in agraph_gt_planes]
            agraph_gt_plane_id = list(set(agraph_gt_planes).intersection(set(agraph_plane_candidates)))[0]
            
            sgraph.remove_edges([(str(node_id), str(sgraph_room_id))])
            sgraph.add_nodes([(str(new_node_id), sgraph.get_attributes_of_node(node_id))])
            sgraph.add_edges([(str(new_node_id), str(sgraph_room_id), {})])
            GT["Plane"].remove([int(agraph_gt_plane_id), int(node_id)])
            GT["Plane"] = GT["Plane"] + [[int(agraph_gt_plane_id), int(new_node_id)]]

    return agraph, sgraph, GT


def create_deviated_graphs(SEs, Rs, Ps, Ds, I):
    dev_dicts_stack = []
    for se in SEs:
        graph_dict_prior_original = parse_full_graph("Prior",f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{se}/A-Graphs/T0")
        graph_dict_online_original = parse_full_graph("Online",f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{se}/S-Graphs/RAll")
        prior_graph_original = GraphWrapper(graph_def = graph_dict_prior_original)
        online_graph_original = GraphWrapper(graph_def = graph_dict_online_original)
        f = open(f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{se}/A-Graphs/T{0}/ground_truth.json")
        gt_matches_original = json.load(f)

        prior_graph_splitted, online_graph_splitted, GT_nd_splitted = divide_shared_planes(prior_graph_original, online_graph_original, gt_matches_original["not_deviated"])
        gt_matches_original["not_deviated"] = GT_nd_splitted

        for i in range(I):
            for r in Rs:
                for P in Ps:
                    online_graph = copy.deepcopy(online_graph_splitted)
                    gt_matches = copy.deepcopy(gt_matches_original)
                    online_graph_all_rooms = list(online_graph.filter_graph_by_node_types(["Finite Room"]).get_nodes_ids())
                    online_graph_selected_rooms = np.random.choice(online_graph_all_rooms, r, replace= False)
                    # online_graph_selected_rooms = ["78", "71"]
                    online_graph.remove_nodes(list(set(online_graph_all_rooms)^set(online_graph_selected_rooms)))
                    online_graph.filterout_unparented_nodes()
                    prior_graph = copy.deepcopy(prior_graph_splitted)
                    online_graph_selected_planes = list(online_graph.filter_graph_by_node_types(["Plane"]).get_nodes_ids())
                    gt_nd_rooms_match = gt_matches["not_deviated"]["Finite Room"]
                    gt_nd_planes_match = gt_matches["not_deviated"]["Plane"]
                    gt_nd_rooms_match = [pair for pair in gt_nd_rooms_match if str(pair[1]) in online_graph_selected_rooms]
                    gt_nd_planes_match = [pair for pair in gt_nd_planes_match if str(pair[1]) in online_graph_selected_planes]
                    gt_matches["not_deviated"]["Finite Room"] = gt_nd_rooms_match
                    gt_matches["not_deviated"]["Plane"] = gt_nd_planes_match

                    for p in range(P):
                        gt_selected_rooms_match = gt_nd_rooms_match[np.random.choice(len(gt_nd_rooms_match), 1, replace= False)[0]]
                        neighbour_planes_prior_id = prior_graph.get_neighbourhood_graph(str(gt_selected_rooms_match[0])).find_nodes_by_attrs({"type": "Plane"})
                        available_planes_prior_id = list(set(np.array(neighbour_planes_prior_id, dtype=np.int32)) & set(np.array(gt_nd_planes_match, dtype=np.int32)[:,0]))
                        selected_plane_prior_id = np.array(available_planes_prior_id)[np.random.choice(len(available_planes_prior_id), 1, replace= False)][0]
                        geo_info = prior_graph.get_attributes_of_node(str(selected_plane_prior_id))["Geometric_info"]
                        point, normal = geo_info[:3], geo_info[3:]
                        d = np.random.uniform(Ds[0], Ds[1]) * random.choice([-1,1])
                        new_point = point + normal*d
                        geo_info = np.concatenate(([ new_point, normal]), axis= 0, dtype = np.float64)
                        prior_graph.set_node_attributes("Geometric_info", {str(selected_plane_prior_id):geo_info})

                        prior_planes_centers = np.stack([prior_graph.get_attributes_of_node(node_id)["Geometric_info"] for node_id in neighbour_planes_prior_id])
                        new_room_center = compute_room_center(prior_planes_centers)
                        prior_graph.set_node_attributes("Geometric_info", {str(gt_selected_rooms_match[0]):new_room_center})

                        deviated_pair_idxes = np.argwhere(np.array(gt_matches["not_deviated"]["Plane"])[:,0] == np.int64(selected_plane_prior_id))
                        if len(deviated_pair_idxes) == 2:
                            candidates_dev_plane_online_id = np.array(gt_matches["not_deviated"]["Plane"])[deviated_pair_idxes,1]
                            neigh_planes_online_ids = online_graph.get_neighbourhood_graph(str(gt_selected_rooms_match[1])).find_nodes_by_attrs({"type": "Plane"})
                            dev_plane_online_id = candidates_dev_plane_online_id[np.where([i in np.array(neigh_planes_online_ids, dtype=np.int32) for i in np.array(candidates_dev_plane_online_id, dtype=np.int32)])[0][0]][0]
                            moving_pair = [int(selected_plane_prior_id), int(dev_plane_online_id)]
                            gt_matches["deviated"]["Plane"].append(moving_pair)
                            gt_matches["not_deviated"]["Plane"].remove(moving_pair)
                        else:
                            gt_matches["deviated"]["Plane"].append(gt_matches["not_deviated"]["Plane"][deviated_pair_idxes[0][0]])
                            gt_matches["not_deviated"]["Plane"].pop(deviated_pair_idxes[0][0])

                    dev_dicts_stack.append({"Prior": prior_graph, "Online": online_graph, "GT": gt_matches})

    return dev_dicts_stack


@timeout_decorator.timeout(200)  # Set the timeout to 300 seconds (5 minutes)
def experiments_stack(dev_dataset,matching_params_comb, base_matching_params, param_space, uniques, line_unique):
    if matching_params_comb:
        matching_params_comb_cp = copy.deepcopy(matching_params_comb)
        base_matching_params_cp = copy.deepcopy(base_matching_params)
        matching_params = match_params_update(matching_params_comb_cp, base_matching_params_cp, param_space)
    else:
        matching_params = copy.deepcopy(base_matching_params)

    scores = Parallel(n_jobs=-1)(delayed(one_experiment)(deviated_graph_dict, matching_params) for deviated_graph_dict in dev_dataset)
    not_null_scores = list(filter(lambda item: item is not None, scores))
    score = np.mean(not_null_scores) * len(not_null_scores) / len(scores)
    uniques.append(len(not_null_scores) / len(scores))
    line_unique.set_data(range(1, len(uniques) + 1), uniques)
    return score


def bayesian_optimization(random_state, iter_num, param_space):
    n_calls = 500
    SEs = [1,2]
    Rs = [2]
    Ps = [1]
    Ds = [0.5,0.15]
    I = 100

    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    line_score, = ax_opt.plot([], [], label=f'Run {iter_num} - score', color = color)
    line_unique, =ax_opt.plot([], [], label=f'Run {iter_num} - unique', linestyle='dashed', color = color)
    uniques = []
    
    deviated_graphs = create_deviated_graphs(SEs, Rs, Ps, Ds, I)
    def objective(matching_params_comb):
        try:
            return -experiments_stack(deviated_graphs, matching_params_comb, matching_params_original, param_space, uniques, line_unique)  # Negative score for minimization
        except timeout_decorator.TimeoutError:
            return 0.0  # Return a very high score if the evaluation times out
    
    def get_nesteddict_value(d, keys):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        return d[keys[-1]]
    
    initial_values = []

    for param_i in param_space:
        mapping = paramter_mapping[param_i.name]
        initial_values.append(get_nesteddict_value(copy.deepcopy(matching_params_original), mapping))
    # initial_values = [0.19639775545292656, 0.3302660870283677, 0.17722970027511514, 0.4435165737690735, 0.6592836339795224, 0.553410484295661, 0.6622208134086656, 0.6027381635820919]

    with tqdm(total= n_calls, desc=f"Run ", position=1, leave=False) as pbar_inner:
        def update_callback(res, scores, iter_num, line_score):
            best_score_index = np.argmin(res.func_vals)
            best_score = -res.func_vals[best_score_index]  # Convert back to positive score
            score = res.func_vals[-1]
            pbar_inner.set_description(f"Best Score: {best_score:.4f}")
            pbar_inner.update(1)
            scores.append(-score)
            line_score.set_data(range(1, len(scores) + 1), scores)
            ax_opt.relim()  # Recompute the limits
            ax_opt.autoscale_view()  # Update the view
            # ax1.relim()  # Recompute the limits
            # ax1.autoscale_view()  # Update the view
            plt.draw()
            plt.pause(0.01)

        scores = []
        ax_opt.legend()  # Update legend after each run
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=param_space,
            n_calls=n_calls,  # Number of iterations
            random_state=random_state,
            # x0=initial_values,
            callback=[lambda res: update_callback(res, scores, iter_num, line_score)]
        )

    # Best parameters and score
    best_params = result.x
    best_score = -result.fun  # Convert back to positive score

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    best_params_full = match_params_update(best_params, matching_params_original, param_space)

    return best_params, best_score, best_params_full

def optimize(target = "unique"):
    best_overall_params = None
    best_overall_score = -np.inf
    n_opimizations = 10

    if target == 'unique':
        param_space = unique_param_space

    with tqdm(total=n_opimizations, desc="Optimization runs", unit="run") as pbar_outer:
        for i in range(n_opimizations):
            random_state = random.randint(0, 10000)
            params, score, params_full = bayesian_optimization(random_state, i + 1, param_space)
            if score > best_overall_score:
                best_overall_params = params
                best_overall_score = score
            pbar_outer.update(1)  # Update the outer progress bar
            plt.savefig(f'{target}_optimization_progress.png')
            plt.savefig(f'./pics/{target}_optimization_progress_{i}.png')
            # Save the best parameters and score to disk
            best_params_dict = {dim.name: best_overall_params[i] for i,dim in enumerate(param_space)}

            with open(f'best_{target}_params_and_score.json', 'w') as f:
                json.dump({
                    'best_params': best_params_dict,
                    'best_score': best_overall_score
                }, f)

            with open(f'best_{target}_params.json', 'w') as f:
                json.dump(params_full, f)


    print("Best Parameters from all runs:", best_overall_params)
    print("Best Score from all runs:", best_overall_score)


    # Save the plot as an image file
    plt.ioff()
    plt.show()

def evaluate(target = "unique"):
    fig_eval, ax_eval = plt.subplots()
    ax_eval.set_xlabel('Deviation length')
    ax_eval.set_ylabel('Score')
    ax_eval.set_title('Evaluation over deviation complexity')

    json_file_path = f"best_{target}_params.json"
    with open(json_file_path) as json_file:
        best_matching_params = json.load(json_file)

    if target == 'unique':
        param_space = unique_param_space


    # R_range = np.arange(0.1,0.4,0.05)
    P_range = np.arange(1,5,1)
    maxD_range = np.arange(0.0,0.8,0.05)
    for p in P_range:
        color = random.choice(list(mcolors.CSS4_COLORS.keys()))
        line_score, = ax_eval.plot([], [], label=f'{p} dev planes - score', color = color)
        line_unique,= ax_eval.plot([], [], label=f'{p} dev planes - unique', linestyle='dashed', color = color)
        uniques = []
        ax_eval.legend()  # Update legend after each run
        scores = []
        xs = []
        for maxd in maxD_range:
            SEs = [1,2]
            Rs = [2]
            Ps = [p]
            Ds = [0.05,maxd]
            I = 50
            
            deviated_graphs = create_deviated_graphs(SEs, Rs, Ps, Ds, I)
            score = experiments_stack(deviated_graphs, None, best_matching_params, param_space, uniques, line_unique)
            scores.append(score)
            xs.append(maxd)
            line_score.set_data(xs, scores)
            ax_eval.relim()  # Recompute the limits
            ax_eval.autoscale_view()  # Update the view
            plt.draw()
            plt.pause(0.01)
            plt.savefig(f'evaluation_{target}_progress.png')




paramter_mapping = { ### MUST MAINTAIN ORDER AS IN matching_params_comb
"inv_point_0_eps": ["invariants", "points", "0", "epsilon"],
"inv_point_0_sig": ["invariants", "points", "0", "sigma"],
"inv_pointnormal_0_sigp": ["invariants", "points&normal", "0", "sigp"],
"inv_pointnormal_0_epsp": ["invariants", "points&normal", "0", "epsp"],
"inv_pointnormal_0_sign": ["invariants", "points&normal", "0", "sign"],
"inv_pointnormal_0_epsn": ["invariants", "points&normal", "0", "epsn"],
"inv_pointnormal_1_sigp": ["invariants", "points&normal", "1", "sigp"],
"inv_pointnormal_1_epsp": ["invariants", "points&normal", "1", "epsp"],
"inv_pointnormal_1_sign": ["invariants", "points&normal", "1", "sign"],
"inv_pointnormal_1_epsn": ["invariants", "points&normal", "1", "epsn"],
"inv_pointnormal_floor_eps": ["invariants", "points&normal", "floor", "epsp"],
"thr_locintra_room": ["thresholds", "local_intralevel", "Finite Room", 0],
"thr_locintra_plane": ["thresholds", "local_intralevel", "Plane", 0],
"thr_locinter_roomplane": ["thresholds", "local_interlevel", "Finite Room - Plane", 0],
"thr_global": ["thresholds", "global", 0],
"db_eps": ["dbscan", "eps"]
}

unique_param_space = [
    # Integer(7, 7, name='solver_iters'),
    Real(0.01, 0.9, name='inv_point_0_eps'),
    Real(0.01, 0.9, name='inv_pointnormal_0_epsp'),
    Real(0.01, 0.9, name='inv_pointnormal_0_sigp'),
    Real(0.01, 0.9, name='inv_pointnormal_0_epsn'),
    Real(0.01, 0.9, name='inv_pointnormal_0_sign'),
    Real(0.01, 0.9, name='inv_pointnormal_1_epsp'),
    Real(0.01, 0.9, name='inv_pointnormal_1_sigp'),
    Real(0.01, 0.9, name='inv_pointnormal_1_epsn'),
    Real(0.01, 0.9, name='inv_pointnormal_1_sign'),
    Real(0.01, 0.9, name='thr_locintra_room'),
    Real(0.01, 0.9, name='thr_locintra_plane'),
    Real(0.01, 0.9, name='thr_locinter_roomplane'),
    Real(0.01, 0.9, name='thr_global'),
    Real(0.001, 0.1, name='db_eps')
]

# matching_params_comb_keys = ["inv_point_0_eps", "inv_point_0_sig","inv_pointnormal_0_sigp", "inv_pointnormal_0_epsp", \
#                              "inv_pointnormal_0_sign", "inv_pointnormal_0_epsn", "inv_pointnormal_1_sigp", "inv_pointnormal_1_epsp", \
#                              "inv_pointnormal_1_sign", "inv_pointnormal_1_epsn", "inv_pointnormal_floor_eps", "thr_locintra_room",\
#                              "thr_locintra_plane", "thr_locinter_roomplane", "thr_global"]
                             
json_file_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/best_unique_params.json"
with open(json_file_path) as json_file:
    matching_params_original = json.load(json_file)


# Set up the plot
plt.ion()
fig_opt, ax_opt = plt.subplots()
ax_opt.set_xlabel('Iteration')
ax_opt.set_ylabel('Best Average Score')
ax_opt.set_title('Bayesian Optimization Progress - Runs')
# fig1, ax1 = plt.subplots()
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('Dataset Scores')
# ax1.set_title('Bayesian Optimization Progress - Datasets')


# optimize()
evaluate()

plt.ioff()
plt.show()