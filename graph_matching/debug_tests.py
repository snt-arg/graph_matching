from Clipper import Clipper
import time
import numpy as np
import math
import matplotlib.pyplot as plt

params = {
    "verbose" : True,
    "invariants": {
        "points&normal": {
            "floor": {
                "sigp": 1000,
                "epsp": 100,
                "sign": 10,
                "epsn": 100
            }
        }
    }
}
# clipper = Clipper("points&normal", "floor", params, None)

# data1 = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64) ### S GRAPHS
# data2 = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64) ### A GRAPHS
# A = np.array([[0,0],[1,1],[2,2]], dtype=np.int32)
# clipper.score_pairwise_consistency(data1, data2, A)
# M, _ = clipper.get_M_C_matrices()
# print(M)


import numpy as np

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def normalize_matric(A):
    for i in range(A.shape[0]):
        A[i] = normalize_vector(A[i])
    return A

def cross_product(v1, v2):
    return np.cross(v1, v2)

def dot_product(v1, v2):
    return np.dot(v1, v2)

def rotation_matrix_between_vectors(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    dot = dot_product(v1, v2)
    cross = cross_product(v1, v2)
    cross_len = np.linalg.norm(cross)

    if cross_len < 1e-6:  # If vectors are parallel
        if dot > 0:  # Vectors are aligned
            return np.identity(3)
        else:  # Vectors are opposite
            # Find a perpendicular vector to v1 to define the rotation axis
            perp_vector = np.array([1, 0, 0]) if abs(v1[0]) < abs(v1[1]) else np.array([0, 1, 0])
            cross = cross_product(v1, perp_vector)
            cross = normalize_vector(cross)
            return 2 * np.outer(cross, cross) - np.identity(3)
    else:
        cos_theta = dot
        sin_theta = cross_len

        axis = normalize_vector(cross)

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.identity(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        return R

def cartesian_to_spherical(v):
    x, y, z = v
    r = np.linalg.norm(v)
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)    # Polar angle
    return r, theta, phi

def rotation_parameters(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    dot = dot_product(v1, v2)
    cross = cross_product(v1, v2)
    cross_len = np.linalg.norm(cross)

    if cross_len < 1e-6:  # If vectors are parallel
        if dot > 0:  # Vectors are aligned
            return 0, (0, 0, 0)  # No rotation
        else:  # Vectors are opposite
            # Any perpendicular vector can be the axis
            return np.pi, (1, 0, 0)
    else:
        angle = np.arccos(dot)
        axis = normalize_vector(cross)
        r, theta, phi = cartesian_to_spherical(axis)
        return angle, (r, theta, phi)

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.6f}" for val in row))

# if __name__ == "__main__":
#     v1 = np.array([1, 0, 0])
#     v2 = np.array([0, -1, 0])

#     rotation_matrix = rotation_matrix_between_vectors(v1, v2)
#     angle, spherical_coords = rotation_parameters(v1, v2)

#     print("Rotation Matrix:")
#     print_matrix(rotation_matrix)
#     print("\nRotation Parameters:")
#     print(f"Angle (radians): {angle}")
#     print(f"Axis (r, theta, phi): {spherical_coords}")

    # print(np.matmul(rotation_matrix, v2))

p1 = np.array([[0,0,0],[1,0,0]])
p2 = np.array([[1,0,0],[0,0,-1]])
p1[1] = normalize_vector(p1[1])
p2[1] = normalize_vector(p2[1])

# ### Normals comparison
# matmul = np.matmul(p1[1],p2[1])
# arccos = np.arccos(matmul)
# print("normals")
# print(matmul)
# print(arccos)

# v_diff = p2[0] - p1[0]
# v_diff = normalize_vector(v_diff)
# cross1 = np.cross(v_diff, p1[1])
# cross2 = np.cross(v_diff, p2[1])
# cross1 = normalize_vector(cross1)
# cross2 = normalize_vector(cross2)
# matmul1 = np.matmul(cross1,cross2)
# arccos1 = np.arccos(matmul1)
# print("with diff")
# print(matmul1)
# print(arccos1)


def vectors_to_spherical(v1, v2):
    # Ensure the vectors are normalize_vectord
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate the angle between the vectors
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Calculate the axis of rotation (cross product)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm != 0:
        axis = axis / axis_norm  # normalize_vector the axis
    else:
        axis = np.array([1, 0, 0])  # If axis is zero, default to x-axis (vectors are parallel)
    
    # Convert axis to spherical coordinates
    x, y, z = axis
    
    # Azimuthal angle (phi) in the x-y plane from the x-axis
    phi = np.arctan2(y, x)
    
    # Polar angle (theta) from the z-axis
    theta = np.arccos(z / np.linalg.norm(axis))
    
    return angle, phi, theta

# Example usage
v1 = np.array([1, 0, 0])
v2 = np.array([1, 0, 0])

# angle, phi, theta = vectors_to_spherical(v1, v2)
# print(f"Rotation Angle: {np.degrees(angle):.2f} degrees")
# print(f"Azimuthal Angle (phi): {np.degrees(phi):.2f} degrees")
# print(f"Polar Angle (theta): {np.degrees(theta):.2f} degrees")


import numpy as np

def kabsch_algorithm(P, Q):
    """
    The Kabsch algorithm: find the optimal rotation matrix that minimizes the
    RMSD between two sets of points P and Q.
    """
    # Compute the covariance matrix
    C = np.dot(np.transpose(P), Q)
    
    # Compute the Singular Value Decomposition
    V, S, W = np.linalg.svd(C)
    
    # Compute the rotation matrix
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    
    U = np.dot(V, W)
    
    return U

def angular_distance(P, Q):
    """ Calculate the angular distance between two sets of vectors """
    dot_products = np.einsum('ij,ij->i', P, Q)
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Ensure values are within valid range
    angles = np.arccos(dot_products)
    return np.degrees(angles)  # Convert radians to degrees if preferred

# Input: two lists of 3D vectors (already normalize_vectord and centered at origin)
# A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Vectors aligned with the x, y, and z axes
# B = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])  # Vectors aligned with the x, y, and z axes

ai = [ 0,1,0,0,-1,0]
aj = [ 1,0,0,0,0,1]

bi = [ 0,1,0,0,-1,0]
bj = [ -1,0,0,0,0,1]


def plot_geometry_set(title, datatype, data, ax = None, color = "red"):  
    if not ax:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    points = np.array(data)[:, :3]
    if datatype == "points&normal":
        normals = np.array(data)[:,3:]
    
    if datatype == "points":
        # Plot the points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label='Points', s = 50, marker = "s")

    # Plot the normals
    elif datatype == "points&normal":
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label='Points')
        for point, normal in zip(points, normals):
            ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.3, arrow_length_ratio=0.1, color='red')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    return points

def plot_geometry_setlist(figure_name, set_list, datatype):
    fig = plt.figure(figure_name, figsize=(10, 7))
    all_points = np.empty((0, 3))
    axs = []
    colors = ["blue", "green"]
    for i_set_list, data in enumerate(set_list):
        plot_number = 100 + 10 * len(set_list) + i_set_list + 1
        ax = fig.add_subplot(plot_number, projection='3d')
        axs.append(ax)
        points = plot_geometry_set(str(i_set_list), datatype, data, ax, colors[i_set_list])
        all_points = np.vstack((all_points, points))

    # all_points = np.vstack((points1, points2))
    x_limits = (all_points[:, 0].min(), all_points[:, 0].max())
    y_limits = (all_points[:, 1].min(), all_points[:, 1].max())
    z_limits = (all_points[:, 2].min(), all_points[:, 2].max())

    for ax in axs:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)


ak = np.array(aj[:3]) - np.array(ai[:3])
bk = np.array(bj[:3]) - np.array(bi[:3])
euclidean_diff = abs(np.linalg.norm(ak) - np.linalg.norm(bk))

A = np.array([ai[-3:], aj[-3:], ak])
B = np.array([bi[-3:], bj[-3:], bk])   
A, B = normalize_matric(A), normalize_matric(B)
# Find the optimal rotation matrix
R = kabsch_algorithm(A, B)

# Apply the rotation to the first set of vectors
A_rotated = np.dot(A, R)

# Calculate the angular distance between the rotated vectors and the second set
angles = angular_distance(A_rotated, B)

# Calculate the mean angular distance
mean_angular_distance = np.mean(angles)

# print("euclidean_diff :\n", euclidean_diff)
# print("Optimal rotation matrix:\n", R)
# print("Angular distances (in degrees):", angles)
# print("Mean angular distance (in degrees):", mean_angular_distance)

sigp = 0.5
epsp = 0.2
sign = 10
epsn = 100
params = {
    "verbose" : True,
    "invariants": {
        "points&normal": {
            "floor": {
                "sigp": 1000,
                "epsp": 100,
                "sign": 10,
                "epsn": 100
            }
        }
    }
}

# mean_angular_distance = 10.
sp = math.exp(-0.5 * euclidean_diff * euclidean_diff / (sigp * sigp))
sp = sp if euclidean_diff < epsp else 0.
sn = math.exp(-0.5 * mean_angular_distance * mean_angular_distance / (sign * sign))
sn = sn if euclidean_diff < epsn else 0.
score = sp * sn
print("sp :", sp, " dn", sn, " score", score)


print("CLPPER CALL")
data1 = np.array([ai,aj], dtype=np.float64) ### S GRAPHS
data2 = np.array([bi,bj], dtype=np.float64) ### A GRAPHS
plot_geometry_setlist("", [data1,data2], "points&normal")
clipper = Clipper("points&normal", "floor", params, None)
A = np.array([[0,0],[1,1]], dtype=np.int32)
clipper.score_pairwise_consistency(data1, data2, A)
M, _ = clipper.get_M_C_matrices()
print(M)
plt.show()