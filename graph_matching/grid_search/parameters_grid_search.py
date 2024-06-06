import numpy as np
import graph_matching
from graph_matching.GraphMatcher import GraphMatcher
from graph_matching.utils import plane_4_params_to_6_params
import os, json
import pandas as pd
import itertools
import copy
import time
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from joblib import Parallel, delayed
# from graph_wrapper.GraphWrapper import GraphWrapper

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
    if len(estimated_match) == 1:
        gt_match_pairs = set([tuple(pair) for pair in gt_match['not_deviated']['Finite Room']] + [tuple(pair) for pair in gt_match['not_deviated']['Plane']])
        estimated_match_pairs = set([(pair['origin_node'], pair['target_node']) for pair in estimated_match[0]])
        # set_gt_match_pairs = set(gt_match_pairs)
        common_tuples = [t for t in estimated_match_pairs if t in gt_match_pairs]
        score = len(common_tuples)/len(gt_match_pairs) - (len(estimated_match_pairs) - len(common_tuples))/len(gt_match_pairs)
    else:
        score = 0.0
    
    return score

def one_experiment(exp_params, matching_params):
    gm = GraphMatcher(PrintLogger(), log_level=0)
    gm.set_parameters(matching_params)
    graph_dict_prior = parse_full_graph("Prior",f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{exp_params['SE']}/A-Graphs/T{exp_params['T']}")
    graph_dict_online = parse_full_graph("Online",f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{exp_params['SE']}/S-Graphs/R{exp_params['R']}")
    gm.set_graph_from_dict(graph_dict_prior, graph_dict_prior["name"])
    gm.set_graph_from_dict(graph_dict_online, graph_dict_online["name"])
    f = open(f"/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/graph_matching/grid_search/graph_logs/SE{exp_params['SE']}/A-Graphs/T{exp_params['T']}/ground_truth.json")
    gt_matches = json.load(f)
    success, matches, matches_full, matches_dev = gm.match("Prior", "Online")
    if matches:
        # for match in matches[0]:
        #     print(f"dbg [{match['origin_node']} , {match['target_node']}] {match['origin_node_attrs']['type']}")
        score = score_estimated_match(matches, gt_matches)
    else:
        score = 0.0
    
    return score

def match_params_update(matching_params_comb):
    json_file_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/config/syntheticDS_params.json"
    with open(json_file_path) as json_file:
        matching_params = json.load(json_file)

    matching_params_comb_keys = ["inv_point_0_eps", "inv_pointnormal_0_eps", "inv_pointnormal_1_eps", "inv_pointnormal_1_eps", \
                                 "thr_locintra_room","thr_locintra_plane", "thr_locinter_roomplane", "thr_global", "solver_iters"]

    paramter_mapping = { ### MUST MAINTAIN ORDER AS IN matching_params_comb
    "inv_point_0_eps": ["invariants", "points", "0", "epsilon"],
    "inv_pointnormal_0_eps": ["invariants", "points&normal", "0", "epsp"],
    "inv_pointnormal_1_eps": ["invariants", "points&normal", "1", "epsp"],
    "inv_pointnormal_floor_eps": ["invariants", "points&normal", "floor", "epsp"],
    "thr_locintra_room": ["thresholds", "local_intralevel", "Finite Room", 0],
    "thr_locintra_plane": ["thresholds", "local_intralevel", "Plane", 0],
    "thr_locinter_roomplane": ["thresholds", "local_interlevel", "Finite Room - Plane", 0],
    "thr_global": ["thresholds", "global", 0],
    "solver_iters": ["solver_iterations"]
    }

    for i, param_key in enumerate(matching_params_comb_keys):
        mapping = paramter_mapping[param_key]

        def update_nested_dict(d, keys, value):
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value

        update_nested_dict(matching_params, mapping, matching_params_comb[i])

    return matching_params
    
def experiments_stack(matching_params_comb):
    matching_params_comb_cp = copy.deepcopy(matching_params_comb)
    matching_params_comb_cp.append(10)
    matching_params = match_params_update(matching_params_comb_cp)

    SEs = [1,2]
    Ts = [0,1,2,4,5]
    Rs = {1:2, 2:3}
    I = 4
    exp_params_stack = []
    for se in SEs:
        for t in Ts:
            r = Rs[se]
            scores = []
            for i in range(I):
                exp_params_stack.append({"SE": se, "T": t, "R": r, "i": i})
    scores = Parallel(n_jobs=-1)(delayed(one_experiment)(exp_params, matching_params) for exp_params in exp_params_stack)
    return np.mean(scores)



def bayesian_optimization(random_state):
    def objective(matching_params_comb):
        return -experiments_stack(matching_params_comb)  # Negative score for minimization
    
    def print_iteration(res):
        n_iter = len(res.func_vals)
        best_score_index = np.argmin(res.func_vals)
        best_score = -res.func_vals[best_score_index]  # Convert back to positive score
        tqdm.write(f"Iteration {n_iter} - Best Score: {best_score:.4f}")

    param_space = [
        # Integer(7, 7, name='solver_iters'),
        Real(0.1, 0.5, name='inv_point_0_eps'),
        Real(0.1, 0.5, name='inv_pointnormal_0_eps'),
        Real(0.1, 0.5, name='inv_pointnormal_1_eps'),
        Real(0.1, 0.5, name='inv_pointnormal_floor_eps'),
        Real(0.1, 0.85, name='thr_locintra_room'),
        Real(0.1, 0.85, name='thr_locintra_plane'),
        Real(0.1, 0.85, name='thr_locinter_roomplane'),
        Real(0.1, 0.85, name='thr_global')
    ]

    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=param_space,
        n_calls=10,  # Number of iterations
        random_state=random_state,
        callback=[print_iteration]
    )

    # Best parameters and score
    best_params = result.x
    best_score = -result.fun  # Convert back to positive score

    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)

    return best_params, best_score

def optimize():
    best_overall_params = None
    best_overall_score = -np.inf

    with tqdm(total=10, desc="Optimization runs", unit="run") as pbar_outer:
        for i in range(10):
            random_state = i  # Different random state for each run
            print(f"Starting optimization run {i + 1} with random state {random_state}")
            params, score = bayesian_optimization(random_state)
            print(f"Run {i + 1} - Best Score: {score:.4f}")
            if score > best_overall_score:
                best_overall_params = params
                best_overall_score = score
            pbar_outer.update(1)  # Update the outer progress bar

    print("Best Parameters from all runs:", best_overall_params)
    print("Best Score from all runs:", best_overall_score)

# def grid_search():
#     # matching_package_path = graph_matching.__file__
#     # json_file_path = os.path.join(matching_package_path[:-11], "config/syntheticDS_params.json")
#     json_file_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_matching/config/syntheticDS_params.json"
#     with open(json_file_path) as json_file:
#         matching_params_original = json.load(json_file)
    
#     parameter_grid = {
#     "inv_point_0_eps": [0.1, 0.25, 0.6],
#     "inv_pointnormal_0_eps": [0.1, 0.25, 0.6],
#     "inv_pointnormal_1_eps": [0.1, 0.25, 0.6],
#     "inv_pointnormal_floor_eps": [0.1, 0.25, 0.6],
#     "thr_locintra_room": [0.2, 0.5, 0.8],
#     "thr_locintra_plane": [0.2, 0.5, 0.8],
#     "thr_locinter_roomplane": [0.2, 0.5, 0.8],
#     "thr_global": [0.2, 0.5, 0.8],
#     "solver_iters": [7],
#     }

#     paramter_mapping = {
#     "inv_point_0_eps": ["invariants", "points", "0", "epsilon"],
#     "inv_pointnormal_0_eps": ["invariants", "points&normal", "0", "epsp"],
#     "inv_pointnormal_1_eps": ["invariants", "points&normal", "1", "epsp"],
#     "inv_pointnormal_floor_eps": ["invariants", "points&normal", "floor", "epsp"],
#     "thr_locintra_room": ["thresholds", "local_intralevel", "Finite Room", 0],
#     "thr_locintra_plane": ["thresholds", "local_intralevel", "Plane", 0],
#     "thr_locinter_roomplane": ["thresholds", "local_interlevel", "Finite Room - Plane", 0],
#     "thr_global": ["thresholds", "global", 0],
#     "solver_iters": ["solver_iterations"]
#     }

#     param_combinations = list(itertools.product(*parameter_grid.values()))
#     # Prepare a list to store the results
#     results = []

#     # Run grid search over each parameter combination
#     for param_combination in tqdm(param_combinations):
#         param_dict = dict(zip(parameter_grid.keys(), param_combination))
#         matching_params = copy.deepcopy(matching_params_original)
#         for param_key in param_dict.keys():
#             mapping = paramter_mapping[param_key]

#             def update_nested_dict(d, keys, value):
#                 for key in keys[:-1]:
#                     d = d.setdefault(key, {})
#                 d[keys[-1]] = value

#             update_nested_dict(matching_params, mapping, param_dict[param_key])

#         score = experiments_stack(matching_params)
        
#         # Store the results
#         result = {**param_dict, "score": score}
#         results.append(result)

#     # Convert results to a pandas DataFrame
#     results_df = pd.DataFrame(results)

#     # Identify the parameter combination with the highest score
#     best_result = results_df.loc[results_df['score'].idxmax()]

#     # Save the DataFrame to a file
#     results_df.to_csv("grid_search_results.csv", index=False)
#     best_result.to_frame().T.to_csv("best_result.csv", index=False)

#     # Display the DataFrame
#     print("Grid Search Results:")
#     print(results_df)
#     print("\nBest Result:")
#     print(best_result.to_frame().T)
optimize()