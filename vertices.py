from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

data = []

with open('data\\vertices.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        data.append(row)

print("Read csv vertex data")
print(data)
print("-------------------")
print("")

algorithm = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
data = StandardScaler().fit_transform(data) * 2
algorithm.fit(data)

y_pred = algorithm.labels_.astype(np.int)

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

ax.scatter(xs, ys, zs, zdir='z', s=2, c=cs, depthshade=False)
plt.show()