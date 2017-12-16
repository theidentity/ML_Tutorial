import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# read
X = np.loadtxt('dummyset')
x,y = X[:,0],X[:,1]

# visualize
plt.plot(x,y,'ro',color='green')
plt.show()

# fit and cluster
model = KMeans(n_clusters=2, random_state=0).fit(X)
Y_pred = model.predict(X)

# display cluster 1
cluster1 = X[np.where(Y_pred==0)]
x,y = cluster1[:,0],cluster1[:,1]
plt.plot(x,y,'ro',color='red')

# display cluster 2
cluster2 = X[np.where(Y_pred==1)]
x,y = cluster2[:,0],cluster2[:,1]
plt.plot(x,y,'ro',color='blue')

plt.show()
