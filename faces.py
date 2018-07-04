from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import face_parser as fp

data = []

filename = "data/faces.obj"

vertices, faces = fp.get_mesh_data(filename)
matrix = fp.build_adjacency_matrix(faces)

#algorithm = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
#algorithm = cluster.SpectralClustering(n_clusters=5, affinity="precomputed")
#matrix = StandardScaler().fit_transform(matrix)
#algorithm.fit(matrix)

#y_pred = algorithm._labels.astype(np.int)

print("\nPerforming spectral clustering")

y_pred = cluster.spectral_clustering(n_clusters=19, affinity=matrix)

print("\nProduced " + str(y_pred.size) + " labels")

#out = ""
#print("Predictions")
#for i in range(y_pred.size):
#    out += str(y_pred[i]) + ", "
#print(out)

#print(data.size)
#print(y_pred.size)

npvertices = np.array(vertices)
xs = npvertices[:,0].tolist()
ys = npvertices[:,1].tolist()
zs = npvertices[:,2].tolist()
cs = y_pred.tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_zlim(-2000, 2000)
ax.scatter(xs, ys, zs, zdir='z', s=2, c=cs, depthshade=False)
plt.show()
