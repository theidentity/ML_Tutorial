import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def read():
	data = np.loadtxt('dummyset')
	return data

def display(X):
	x,y = X[:,0],X[:,1]
	plt.plot(x,y,'ro',color='green')
	plt.show()

def fit_model(X,clusters=2):
	model = KMeans(n_clusters=clusters, random_state=0).fit(X)
	return model

def predict(X,model):
	Y_pred = model.predict(X)
	return X,Y_pred

def display_result(X,Y):
	cluster1 = X[np.where(Y==0)]
	cluster2 = X[np.where(Y==1)]
	plt.plot(cluster1[:,0],cluster1[:,1],'ro',color='red')
	plt.plot(cluster2[:,0],cluster2[:,1],'ro',color='blue')
	plt.show()


X = read()
display(X)

model = fit_model(X,2)
X,Y_pred = predict(X,model)

display_result(X,Y_pred)