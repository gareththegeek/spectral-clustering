import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import obj_parser as op

filename = "data/lowpolycity.obj"

vertices, faces = op.get_mesh_data(filename)
normals = op.build_face_normals(vertices, faces)

f = lambda v: v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

print("\nCalculating distances from origin")
magnitudes = np.fromiter((f(v) for v in vertices), np.float, count=len(vertices))

data = np.column_stack((vertices, normals))
data = np.column_stack((data, magnitudes))
#data = magnitudes.reshape(-1, 1)
#data = normals
print("\nCombined data set")
print(data.astype(np.float))

print("\nPerforming spectral clustering")

algorithm = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity="rbf")
data = StandardScaler().fit_transform(data)
algorithm.fit(data)

y_pred = algorithm.labels_.astype(np.int)

print("\nGenerated " + str(len(y_pred)) + " labels")

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
