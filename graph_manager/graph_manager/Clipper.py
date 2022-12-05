import  clipperpy
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
            iparams.sigp = 0.5
            iparams.epsp = 0.5
            iparams.sign = 0.10
            iparams.epsn = 0.35
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
        return(C, M)


    def mix_C_matrices(self, external_C):
        original_C = self.clipper.get_constraint_matrix()
        M = self.clipper.get_affinity_matrix() 
        C = np.multiply(original_C, external_C)
        self.clipper.set_matrix_data(M, C)


    def exagerate_constraint_matrix_value(self, position):
        C = self.clipper.get_constraint_matrix()
        M = self.clipper.get_affinity_matrix()
        C[position[0],position[1]] = np.sum(C) + 1
        self.clipper.set_matrix_data(M, C)

    def solve_clipper(self):
        t0 = time.perf_counter()
        self.clipper.solve()
        t1 = time.perf_counter()
        Ain = self.clipper.get_selected_associations()
        return(Ain)


    def categorize_clipper_output(self, Ain_numerical, nodes1, nodes2):
        Ain_categorical = np.array([[nodes1[pair[0]],nodes2[pair[1]]] for pair in Ain_numerical])
        return Ain_categorical



