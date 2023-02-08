import clipperpy
import time
import numpy as np


class Clipper():
    def __init__(self, data_type, name, logger = None) -> None:
        self.logger = logger
        invariant = self.stored_invriants(data_type, name)
        self.create_clipper_object(invariant)

    def stored_invriants(self, data_type, name = None):
        if data_type == "points":
            if name == 1:
                iparams = clipperpy.invariants.EuclideanDistanceParams()
                iparams.sigma = 0.1 # 0.015
                iparams.epsilon = 0.6 # 0.02
                iparams.mindist = 0 
                invariant = clipperpy.invariants.EuclideanDistance(iparams)
            else:
                iparams = clipperpy.invariants.EuclideanDistanceParams()
                iparams.sigma = 0.01     # spread / variance of exponential kernel
                iparams.epsilon = 0.06   # bound on consistency score, determines if inlier/outlier
                iparams.mindist = 0      # minimum allowable distance between inlier points in the same dataset
                invariant = clipperpy.invariants.EuclideanDistance(iparams)
        elif data_type == "points&normal":
            if name == 1:
                iparams = clipperpy.invariants.PointNormalDistanceParams()
                iparams.sigp = 0.5     # point - spread of exp kernel
                iparams.epsp = 0.5     # point - bound on consistency score
                iparams.sign = 0.10    # normal - spread of exp kernel
                iparams.epsn = 0.35    # normal - bound on consistency score
                invariant = clipperpy.invariants.PointNormalDistance(iparams)
            elif name == 2:
                iparams = clipperpy.invariants.PointNormalDistanceParams()
                iparams.sigp = 0.4     # point - spread of exp kernel
                iparams.epsp = 0.4     # point - bound on consistency score
                iparams.sign = 0.08    # normal - spread of exp kernel
                iparams.epsn = 0.25    # normal - bound on consistency score
                invariant = clipperpy.invariants.PointNormalDistance(iparams)
            else:
                iparams = clipperpy.invariants.PointNormalDistanceParams()
                iparams.sigp = 0.5     # point - spread of exp kernel
                iparams.epsp = 0.5     # point - bound on consistency score
                iparams.sign = 0.10    # normal - spread of exp kernel
                iparams.epsn = 0.35    # normal - bound on consistency score
                invariant = clipperpy.invariants.PointNormalDistance(iparams)


        return invariant



    def create_clipper_object(self, invariant):
        params = clipperpy.Params()
        self.clipper = clipperpy.CLIPPER(invariant, params)
        

    def score_pairwise_consistency(self, D1, D2, A):
        t0 = time.perf_counter()
        self.A = A
        self.clipper.score_pairwise_consistency(D1.T, D2.T, A)
        C = self.get_constraint_matrix()
        M = self.get_affinity_matrix()
        t1 = time.perf_counter()
        print(f"Affinity matrix creation took {t1-t0:.3f} seconds")


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
        t0 = time.perf_counter()
        self.clipper.solve()
        t1 = time.perf_counter()
        len_solution = len(self.clipper.get_solution().nodes)
        if len_solution > 0:
            # n_non_diagonal_entries_solution = len_solution*(len_solution-1)
            n_non_diagonal_entries_solution = len_solution
            if not n_non_diagonal_entries_solution:
                n_non_diagonal_entries_solution = 1
            avg_score = self.clipper.get_solution().score / n_non_diagonal_entries_solution
            # self.logger.info("n_non_diagonal_entries_solution {}".format(n_non_diagonal_entries_solution))
            # self.logger.info("self.get_affinity_matrix() {}".format(self.get_affinity_matrix()))
            # self.logger.info("self.clipper.get_solution().u {}".format(self.clipper.get_solution().u))
            # self.logger.info("self.clipper.get_solution().score {}".format(self.clipper.get_solution().score))
            # self.logger.info("avg_score {}".format(avg_score))
        else:
            avg_score = 0

        return(self.clipper.get_selected_associations(), avg_score)


    def categorize_clipper_output(self, Ain_numerical, nodes1, nodes2):
        Ain_categorical = set([(nodes1[pair[0]],nodes2[pair[1]]) for pair in Ain_numerical])
        return Ain_categorical


    def get_M_C_matrices(self):
        C = self.get_constraint_matrix()
        M = self.get_affinity_matrix()
        return(M,C)


    def get_score_all_inital_u(self):
        M = self.get_affinity_matrix()
        len_u = M.shape[0]
        # self.logger.info("A {}".format(self.A))
        # self.logger.info("M_aux {}".format(M))
        u = np.ones(len_u)
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