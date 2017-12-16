import numpy as np
import matplotlib.pyplot as plt


def read():
	data = np.loadtxt('dummyset')
	return data

def display(X):
	x,y = X[:,0],X[:,1]
	plt.plot(x,y,'ro',color='green')
	plt.show()

def get_centroid(cluster):
	x = np.mean(cluster[:,0])
	y = np.mean(cluster[:,1])
	return x,y

def reassign_cluster(X,c1,c2):
	diff_sqr = np.power(X-c1,2)
	d1 = np.sum(diff_sqr,axis=1)
	diff_sqr = np.power(X-c2,2)
	d2 = np.sum(diff_sqr,axis=1)
	Y = d1>=d2
	return Y

def predict(X):
	l = X.shape[0]
	Y = np.random.randint(2, size=l)
	for i in range(2):
		display_result(X,Y)
		cluster1 = X[np.where(Y==0)] 
		cluster2 = X[np.where(Y==1)] 
		centroid1 = get_centroid(cluster1)
		centroid2 = get_centroid(cluster2)
		Y = reassign_cluster(X,centroid1,centroid2)
	return X,Y

def display_result(X,Y):
	cluster1 = X[np.where(Y==0)]
	cluster2 = X[np.where(Y==1)]
	plt.plot(cluster1[:,0],cluster1[:,1],'ro',color='red')
	plt.plot(cluster2[:,0],cluster2[:,1],'ro',color='blue')
	plt.show()


X = read()
display(X)

X,Y_pred = predict(X)
