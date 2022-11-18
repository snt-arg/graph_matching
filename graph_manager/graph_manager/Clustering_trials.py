import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import pandas as pd


dataset = []

def generateRandomRoom():
    distances = np.random.uniform(0,10,2)
    rotation = np.random.uniform(0,3.14)
    position = np.random.uniform(0,10,2)

    walls = position + distances


def generateGridBuildingWalls():

    grid_dimensions = [3,3]
    spacing = np.array([4,8])
    wall_thickness = 0.4

    normals = np.array([[0,1],[-1,0],[0,-1],[1,0]])
    reduced_spacing = np.multiply(normals,np.tile((spacing - wall_thickness)/2,(4,1)))

    surfaces = np.array([])
    room_tag = 0

    for row in range(grid_dimensions[0]):
        for column in range(grid_dimensions[1]):
            center = np.multiply(np.array([row,column]), spacing)
            new_surfaces = np.tile(center,(4,1)) + reduced_spacing
            new_surfaces = np.append(new_surfaces, -normals, axis=1)
            new_surfaces = np.append(new_surfaces, np.ones((4,1))*room_tag, axis=1)
            if surfaces.size == 0:
                surfaces = new_surfaces
            else:
                surfaces = np.concatenate((surfaces, new_surfaces), axis= 0 )

            room_tag += 1

    # np.random.shuffle(surfaces)
    return surfaces


def plotSurfaces(surfaces):
    df = pd.DataFrame(surfaces)
    groups = df.groupby(4)
    for name, group in groups:
        plt.plot(group[0], group[1], marker='o', linestyle='', markersize=7, label=name)
    plt.legend()
    normals_x = np.append(np.expand_dims(surfaces[:,0], axis=1), np.expand_dims(surfaces[:,0] + surfaces[:,2], axis=1), axis=1)
    normals_y = np.append(np.expand_dims(surfaces[:,1], axis=1), np.expand_dims(surfaces[:,1] + surfaces[:,3], axis=1), axis=1)
    for i in range(normals_x.shape[0]):
        plt.plot(normals_x[i,:], normals_y[i,:], color='b')
    plt.show()


def computeKMeans(data):
    kmeans = KMeans(random_state=0).fit(data[:,:4])
    labeled_data = np.append(data, np.expand_dims(kmeans.labels_, axis=1), axis=1)
    plotSurfaces(labeled_data)
    return kmeans.labels_

# surfaces = generateGridBuildingWalls()
# # plotSurfaces(surfaces)
# computeKMeans(surfaces[:,:-1])

x = np.array([[1,1,1], [2,2,2], [3,3,3]])
y = np.array([[4,4,4], [5,5,5], [6,6,6]])

c = np.array([[[4,4,4],[4,4,4]], [[4,4,4],[5,5,5]], [[4,4,4],[6,6,6]]])

print(np.stack((x, y), axis=1))