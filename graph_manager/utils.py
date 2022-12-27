import numpy as np

def plane_4_params_to_6_params(plane):
    point = np.array(plane[:3])*plane[3]
    return(np.concatenate((point, plane[:3])))