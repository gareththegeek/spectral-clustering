from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import obj_parser as op

data = []

filename = "data/lowpolycity.obj"

vertices, faces = op.get_mesh_data(filename)
matrix = op.build_adjacency_matrix(faces)

print("\nPerforming spectral clustering")

y_pred = cluster.spectral_clustering(n_clusters=6, affinity=matrix)

print("\nProduced " + str(y_pred.size) + " labels")

npvertices = np.array(vertices)
xs = npvertices[:,0].tolist()
ys = npvertices[:,1].tolist()
zs = npvertices[:,2].tolist()
cs = y_pred.tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 4)
ax.scatter(xs, ys, zs, zdir='y', s=2, c=cs, depthshade=False)
plt.show()
