import numpy as np

def plane_4_params_to_6_params(plane):
    point = -plane[3]*np.array(plane[:3])
    return(np.concatenate((point, plane[:3])))