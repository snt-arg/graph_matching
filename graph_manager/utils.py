import numpy as np
import time

def plane_4_params_to_6_params(plane):
    point = -plane[3]*np.array(plane[:3])
    return(np.concatenate((point, plane[:3])))


def plane_6_params_to_4_params(point_and_normal):
    normal = np.array(point_and_normal[3:6])
    distance = - np.linalg.norm(point_and_normal[:3]) * np.sign(np.dot(normal, point_and_normal[:3]))
    return(np.concatenate((normal, [distance])))


def translate_plane_definition(points_and_normals, translation, rotation, logger):
    translated_points_and_normals = []
    for point_and_normal in points_and_normals:
        normal_and_distance = plane_6_params_to_4_params(point_and_normal)
        print(normal_and_distance)
        translated_normal_and_distance = translate_normal_and_distance(normal_and_distance, translation, rotation, logger)
        print(translated_normal_and_distance)
        translated_point_and_normal = plane_4_params_to_6_params(translated_normal_and_distance)
        print(translated_point_and_normal)
        translated_points_and_normals.append(translated_point_and_normal)
    return np.array(translated_points_and_normals)


def translate_normal_and_distance(original, translation, rotation, logger):
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
    transformed_normalized = transformed / np.sqrt(np.power(transformed[:3],2).sum(axis=0))
    # logger.info("transformed_normalized - {}".format(transformed_normalized))
    # logger.info("np.transpose(transformed_normalized) - {}".format(np.transpose(transformed_normalized)))
    # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
    return transformed_normalized


# pn = np.array([[2,0,0,1,0,0]])
# tra = np.array([1,0,0])
# rot = np.array([[1,0,0],[0,1,0],[0,0,1]])


# translate_plane_definition(pn, tra, rot, None)


