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

data = StandardScaler().fit_transform(data)

npdata = np.array(data)
xs = npdata[:,0].tolist()
ys = npdata[:,1].tolist()
zs = npdata[:,2].tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, zdir='y', s=2, c=None, depthshade=False)
plt.show()