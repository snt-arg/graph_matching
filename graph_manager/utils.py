import numpy as np
import time
import math

def plane_4_params_to_6_params(plane):
    normal = plane[:3]
    distance = plane[3]
    point = -distance*np.array(normal)
    return(np.concatenate((point, normal)))


def plane_6_params_to_4_params(point_and_normal):
    point = np.array(point_and_normal[:3])
    normal = np.array(point_and_normal[3:6])
    distance = - np.dot(point, normal)
    return(np.concatenate((normal, [distance])))


def transform_plane_definition(points_and_normals, translation, rotation, logger = None):
    translated_points_and_normals = []
    for point_and_normal in points_and_normals:
        normal_and_distance = plane_6_params_to_4_params(point_and_normal)
        print(normal_and_distance)
        translated_normal_and_distance = transform_normal_and_distance(normal_and_distance, translation, rotation, logger)
        print(translated_normal_and_distance)
        translated_point_and_normal = plane_4_params_to_6_params(translated_normal_and_distance)
        print(translated_point_and_normal)
        translated_points_and_normals.append(translated_point_and_normal)
    return np.array(translated_points_and_normals)


def transform_normal_and_distance(original, translation, rotation, logger = None):
    # start_time = time.time()
    #### Build transform matrix
    # logger.info("original - {}".format(original))
    rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
    translation_1 = np.array([np.concatenate((-translation, np.array([1.0])), axis=0)])
    full_transformation_matrix = np.concatenate((rotation_0, translation_1), axis=0)
    # logger.info("full_transformation_matrix - {}".format(full_transformation_matrix))

    #### Matrix multiplication
    transformed = np.transpose(np.matmul(full_transformation_matrix,original))
    # logger.info("transformed - {}".format(transformed))
    # transformed = np.array([2,0,0,6])
    print("transformed", transformed)
    normalization = np.sqrt(np.power(transformed[:3],2).sum(axis=0))
    transformed_normalized = np.concatenate((transformed[:3] / normalization, [transformed[3] * normalization]))
    # transformed_normalized = transformed
    print("transformed_normalized", transformed_normalized)
    # logger.info("transformed_normalized - {}".format(transformed_normalized))
    # logger.info("np.transpose(transformed_normalized) - {}".format(np.transpose(transformed_normalized)))
    # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
    return transformed_normalized


def transform_point(original, translation, rotation):
    original = np.array(original)
    first_group = np.concatenate((rotation, np.expand_dims( translation, axis=1)), axis=1)
    second_group = [np.array([0.,0.,0.,1.0])]
    full_transformation_matrix = np.concatenate((first_group, second_group), axis=0)
    tmp = np.empty((original.shape[0],original.shape[1]+1),dtype=original.dtype)
    tmp[:,0:3] = original
    tmp[:,3] = 1
    return full_transformation_matrix.dot(tmp.transpose())[0:3].transpose() 


def rotation_matrix_from_euler_degrees(phi, theta, psi):
        def Rx(phi):
            return np.matrix([[ 1, 0           , 0           ],\
                        [ 0, math.cos(phi),-math.sin(phi)],\
                        [ 0, math.sin(phi), math.cos(phi)]])
        
        def Ry(theta):
            return np.matrix([[ math.cos(theta), 0, math.sin(theta)],\
                        [ 0           , 1, 0           ],\
                        [-math.sin(theta), 0, math.cos(theta)]])
        
        def Rz(psi):
            return np.matrix([[ math.cos(psi), -math.sin(psi), 0 ],\
                        [ math.sin(psi), math.cos(psi) , 0 ],\
                        [ 0           , 0            , 1 ]])

        def degrees_to_radians(deg):
            return deg*math.pi/180

        R = Rz(degrees_to_radians(psi)) * Ry(degrees_to_radians(theta)) * Rx(degrees_to_radians(phi))
        return np.array(R)

# pn = np.array([[2,0,0,1,0,0]])
# p = np.array([[1,0,0]])
# tra = np.array([1,0,0])
# rot = rotation_matrix_from_euler_degrees(0,0,0)

# print(transform_plane_definition(pn, tra, rot, None))


# print(transform_point(p, -tra, rot))

# [[0.   0.92495263 0.88753341]
# [0.92495263  0.         0.9686793 ]
# [0.88753341 0.9686793  0.        ]]


# print(((0.92495263 + 0.88753341 + 0.9686793)*2 + 3) / 3)
# 2.5955923427238172