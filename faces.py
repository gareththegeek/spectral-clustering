from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import face_parser as fp

data = []

filename = "data/faces.obj"

vertices, matrix = fp.build_adjacency_matrix(filename)

#algorithm = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
#algorithm = cluster.SpectralClustering(n_clusters=5, affinity="precomputed")
#matrix = StandardScaler().fit_transform(matrix)
#algorithm.fit(matrix)

#y_pred = algorithm._labels.astype(np.int)

print("\nPerforming spectral clustering")

y_pred = cluster.spectral_clustering(n_clusters=5, affinity=matrix)

print(y_pred)
print(y_pred.size)
exit()

#out = ""
#print("Predictions")
#for i in range(y_pred.size):
#    out += str(y_pred[i]) + ", "
#print(out)

#print(data.size)
#print(y_pred.size)

npdata = np.array(data)
xs = npdata[:,0].tolist()
ys = npdata[:,1].tolist()
zs = npdata[:,2].tolist()
cs = y_pred.tolist()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, zdir='z', s=20, c=cs, depthshade=False)
plt.show()