import clipperpy
import time
import numpy as np
import json
import os
import pathlib



class Clipper():
    def __init__(self, data_type, name, params, logger = None) -> None:
        self.logger = logger
        self.params = params
        self.invariant = self.stored_invriants(data_type, name)
        self.create_clipper_object(self.invariant)


    def stored_invriants(self, data_type, index = None):
        if data_type == "points":
            iparams = clipperpy.invariants.EuclideanDistanceParams()
            iparams.sigma = self.params["invariants"][data_type][index]["sigma"]
            iparams.epsilon = self.params["invariants"][data_type][index]["epsilon"]
            iparams.mindist = self.params["invariants"][data_type][index]["epsilon"]
            invariant = clipperpy.invariants.EuclideanDistance(iparams)

        elif data_type == "points&normal":
            iparams = clipperpy.invariants.PointNormalDistanceParams()
            iparams.sigp = self.params["invariants"][data_type][index]["sigp"]
            iparams.epsp = self.params["invariants"][data_type][index]["epsp"]
            iparams.sign = self.params["invariants"][data_type][index]["sign"]
            iparams.epsn = self.params["invariants"][data_type][index]["epsn"]
            invariant = clipperpy.invariants.PointNormalDistance(iparams)


        return invariant



    def create_clipper_object(self, invariant):
        params = clipperpy.Params()
        self.clipper = clipperpy.CLIPPER(invariant, params)
        

    def score_pairwise_consistency(self, D1, D2, A):
        t0 = time.perf_counter()
        self.D1, self.D2, self.A = D1, D2, A
        self.clipper.score_pairwise_consistency(D1.T, D2.T, A)
        C = self.get_constraint_matrix()
        M = self.get_affinity_matrix()
        # self.logger.info(f"dbg clipper score_pairwise_consistency C {C}")
        # self.logger.info(f"dbg clipper score_pairwise_consistency M {M}")
        t1 = time.perf_counter()
        # print(f"Affinity matrix creation took {t1-t0:.3f} seconds")
        return C, M


    # def filter_C_M_matrices(self, external_C):
    #     original_C = self.get_constraint_matrix()
    #     original_C += np.eye(original_C.shape[0])
    #     C = np.multiply(original_C, external_C)
    #     original_M = self.get_affinity_matrix()
    #     original_M += np.eye(original_M.shape[0])
    #     M = np.multiply(original_M, external_C)
    #     # print("original_C", original_C)
    #     # print("external_C", external_C)
    #     # print("C", C)
    #     # print(original_C == external_C)
    #     self.clipper.set_matrix_data(M, C)


    # def remove_last_assoc_M_C_matrices(self):
    #     C = self.get_constraint_matrix()
    #     M = self.get_affinity_matrix()
    #     M = M[:-1, :-1]
    #     C = C[:-1, :-1]
    #     self.clipper.set_matrix_data(M, C)


    # def set_M_diagonal_values(self, diagonal_values):
    #     C = self.get_constraint_matrix()
    #     # C += np.eye(C.shape[0])
    #     M = self.get_affinity_matrix()
    #     # np.fill_diagonal(M, diagonal_values)
    #     # M += np.eye(M.shape[0])
    #     self.clipper.set_matrix_data(M, C)


    def solve_clipper(self):
        best_score, best_selected_associations = 0, None
        for i in range(self.params["solver_iterations"]):
            self.create_clipper_object(self.invariant)
            self.score_pairwise_consistency(self.D1, self.D2, self.A)
            t0 = time.perf_counter()
            self.clipper.solve()
            t1 = time.perf_counter()
            solution = self.clipper.get_solution()
            len_solution = len(solution.nodes)
            # self.logger.info(f"dbg clipper solution dir {solution.__dir__()}")
            if len_solution > 0:
                # n_non_diagonal_entries_solution = len_solution*(len_solution-1)
                n_non_diagonal_entries_solution = len_solution
                if not n_non_diagonal_entries_solution:
                    n_non_diagonal_entries_solution = 1
                avg_score = solution.score / n_non_diagonal_entries_solution
                # self.logger.info("n_non_diagonal_entries_solution {}".format(n_non_diagonal_entries_solution))
                # self.logger.info("self.get_affinity_matrix() {}".format(self.get_affinity_matrix()))
                # self.logger.info("self.clipper.get_solution().u {}".format(self.clipper.get_solution().u))
                # self.logger.info("self.clipper.get_solution().score {}".format(self.clipper.get_solution().score))
                # self.logger.info("avg_score {}".format(avg_score))
            else:
                avg_score = 0

            if avg_score > best_score:
                best_score = avg_score
                best_selected_associations = self.clipper.get_selected_associations()
        # self.logger.info(f"dbg clipper solution u {solution.u}")
        # self.logger.info(f"dbg clipper solution score {solution.score}")
        # self.logger.info(f"dbg clipper solution nodes {solution.nodes}")
        # self.logger.info(f"dbg clipper clipper selected {self.clipper.get_selected_associations()}")
        return(best_selected_associations, best_score)


    def categorize_clipper_output(self, Ain_numerical, nodes1, nodes2):
        Ain_categorical = set([(nodes1[pair[0]],nodes2[pair[1]]) for pair in Ain_numerical])
        return Ain_categorical


    def get_M_C_matrices(self):
        C = self.get_constraint_matrix()
        M = self.get_affinity_matrix()
        return(M,C)
    

    def get_score_all_inital_u(self):
        M = self.get_affinity_matrix()
        self.logger.info(f"dbg get_score_all_inital_u M {M}")
        len_u = M.shape[0]
        # self.logger.info("M {}".format(M))
        u = np.ones(len_u)
        # self.logger.info("len_u {}".format(len_u))
        # self.logger.info("u {}".format(u))
        consistency = np.matmul(np.matmul(u,M),u.T)
        # self.logger.info("consistency {}".format(consistency))
        n_non_diagonal_entries = len_u*(len_u -1 )
        # n_non_diagonal_entries = len_u
        consistency_avg = consistency / n_non_diagonal_entries
        # self.logger.info("consistency_avg {}".format(consistency_avg))
        return consistency_avg


    def get_affinity_matrix(self):
        M = self.clipper.get_affinity_matrix()
        M -= np.eye(M.shape[0])
        return M

    def get_constraint_matrix(self):
        C = self.clipper.get_constraint_matrix()
        C -= np.eye(C.shape[0])
        return C