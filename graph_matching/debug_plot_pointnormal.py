import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Example set of points and normals in 3D for the first set
points1 = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 0],
    [0, 2, 2]
])

normals1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
])

# Example set of points and normals in 3D for the second set
points2 = np.array([
    [0, 0, 1],
    [1, -1, 1],
    [-2, 2, 0],
    [0, -2, -2]
])

normals2 = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-1, -1, 0]
])

# Create a 3D plot
fig = plt.figure(figsize=(15, 7))

# Plot for the first set
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='blue', label='Points Set 1')
for point, normal in zip(points1, normals1):
    ax1.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.5, color='red')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Points and Normals Set 1')
ax1.legend()

# Plot for the second set
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='green', label='Points Set 2')
for point, normal in zip(points2, normals2):
    ax2.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.5, color='purple')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Points and Normals Set 2')
ax2.legend()

# Set the same limits for both plots
all_points = np.vstack((points1, points2))
x_limits = (all_points[:, 0].min(), all_points[:, 0].max())
y_limits = (all_points[:, 1].min(), all_points[:, 1].max())
z_limits = (all_points[:, 2].min(), all_points[:, 2].max())

ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

ax2.set_xlim(x_limits)
ax2.set_ylim(y_limits)
ax2.set_zlim(z_limits)

plt.show()
