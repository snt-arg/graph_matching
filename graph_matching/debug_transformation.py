import numpy as np

def compute_transformation(points_a, normals_a, points_b, normals_b):
    # Compute the centroids of both sets
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # Translate points to align centroids with the origin
    points_a_centered = points_a - centroid_a
    points_b_centered = points_b - centroid_b

    # Compute the optimal rotation matrix using Singular Value Decomposition (SVD)
    H = np.dot(points_a_centered.T, points_b_centered)
    U, S, Vt = np.linalg.svd(H)
    rotation_matrix = np.dot(Vt.T, U.T)

    # Ensure the rotation matrix is proper (det(rotation) should be 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # Apply the rotation matrix to the normals as well
    normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

    # Check for reflection by comparing normals
    reflection_needed = False
    for normal_a_transformed, normal_b in zip(normals_a_transformed, normals_b):
        if np.dot(normal_a_transformed, normal_b) < 0:
            reflection_needed = True
            break

    # If reflection is needed, apply it to the rotation matrix
    if reflection_needed:
        reflection_matrix = np.diag([1, 1, -1])
        rotation_matrix = np.dot(rotation_matrix, reflection_matrix)
        normals_a_transformed = np.dot(normals_a, rotation_matrix.T)

    # Compute the translation vector
    translation_vector = centroid_b - np.dot(centroid_a, rotation_matrix.T)

    return translation_vector, rotation_matrix, reflection_needed

def transform_set(points, normals, translation, rotation_matrix):
    # Apply the translation and rotation to each point and normal
    transformed_points = np.dot(points, rotation_matrix.T) + translation
    transformed_normals = np.dot(normals, rotation_matrix.T)
    return transformed_points, transformed_normals

# Example usage
points_a = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]])
normals_a = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1]])
points_b = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]])
normals_b = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1]])

translation, rotation_matrix, reflection_needed = compute_transformation(points_a, normals_a, points_b, normals_b)
transformed_points, transformed_normals = transform_set(points_a, normals_a, translation, rotation_matrix)

print("Translation Vector:\n", translation)
print("Rotation Matrix:\n", rotation_matrix)
print("Reflection Needed:", reflection_needed)
print("Transformed Points:\n", transformed_points)
print("Transformed Normals:\n", transformed_normals)
