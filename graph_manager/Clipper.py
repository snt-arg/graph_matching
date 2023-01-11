import clipperpy
import time
import numpy as np


class Clipper():
    def __init__(self, data_type) -> None:
        invariant = self.define_clipper_invariant(data_type)
        self.create_clipper_object(invariant)

    def define_clipper_invariant(self, data_type):
        if data_type == "points":
            iparams = clipperpy.invariants.EuclideanDistanceParams()
            iparams.sigma = 0.015
            iparams.epsilon = 0.02
            invariant = clipperpy.invariants.EuclideanDistance(iparams)
        elif data_type == "points&normal":
            iparams = clipperpy.invariants.PointNormalDistanceParams()
            iparams.sigp = 0.5 # 0.5
            iparams.epsp = 0.5 # 0.5
            iparams.sign = 0.10 # 0.10
            iparams.epsn = 0.35 # 0.35
            invariant = clipperpy.invariants.PointNormalDistance(iparams)

        return invariant


    def create_clipper_object(self, invariant):
        params = clipperpy.Params()
        self.clipper = clipperpy.CLIPPER(invariant, params)
        

    def score_pairwise_consistency(self, D1, D2, A):
        t0 = time.perf_counter()
        self.clipper.score_pairwise_consistency(D1.T, D2.T, A)
        C = self.clipper.get_constraint_matrix()
        M = self.clipper.get_affinity_matrix()
        t1 = time.perf_counter()
        print(f"Affinity matrix creation took {t1-t0:.3f} seconds")


    def filter_C_M_matrices(self, external_C):
        original_C = self.clipper.get_constraint_matrix()
        original_C += np.eye(original_C.shape[0])
        C = np.multiply(original_C, external_C)
        original_M = self.clipper.get_affinity_matrix()
        original_M += np.eye(original_M.shape[0])
        M = np.multiply(original_M, external_C)
        # print("original_C", original_C)
        # print("external_C", external_C)
        # print("C", C)
        # print(original_C == external_C)
        self.clipper.set_matrix_data(M, C)


    def remove_last_assoc_M_C_matrices(self):
        C = self.clipper.get_constraint_matrix()
        M = self.clipper.get_affinity_matrix()
        M = M[:-1, :-1]
        C = C[:-1, :-1]
        self.clipper.set_matrix_data(M, C)


    def set_M_diagonal_values(self, diagonal_values):
        C = self.clipper.get_constraint_matrix()
        # C += np.eye(C.shape[0])
        M = self.clipper.get_affinity_matrix()
        # np.fill_diagonal(M, diagonal_values)
        # M += np.eye(M.shape[0])
        self.clipper.set_matrix_data(M, C)


    def solve_clipper(self):
        t0 = time.perf_counter()
        self.clipper.solve()
        t1 = time.perf_counter()
        if len(self.clipper.get_solution().nodes) > 0:
            avg_score = self.clipper.get_solution().score / len(self.clipper.get_solution().nodes)
        else:
            avg_score = 0

        return(self.clipper.get_selected_associations(), avg_score)


    def categorize_clipper_output(self, Ain_numerical, nodes1, nodes2):
        Ain_categorical = set([(nodes1[pair[0]],nodes2[pair[1]]) for pair in Ain_numerical])
        return Ain_categorical


    def get_M_C_matrices(self):
        C = self.clipper.get_constraint_matrix()
        M = self.clipper.get_affinity_matrix()
        return(M,C)