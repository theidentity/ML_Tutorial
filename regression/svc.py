import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def read():
	X = np.arange(0,4*3.14,0.1)
	Y = np.sin(X)
	# data = np.loadtxt("pressure")
	# X = data[:,2]
	# Y = data[:,3]
	return X,Y

def displayXY(X,Y):
	plt.plot(X,Y)
	plt.show()

def fitSVC(X,Y):
	X = X.reshape(-1,1)
	Y = Y.reshape(-1,1).ravel()
	model = SVR(kernel='rbf')
	# model = SVR(kernel='linear')
	model.fit(X,Y)
	return model

def predict(X,model):
	X = X.reshape(-1,1)
	Y_pred = model.predict(X)
	return Y_pred

def displayRegressionLine(X,Y,Y_pred):
	plt.plot(X,Y,'ro',color='green')
	plt.plot(X,Y_pred,color='blue')
	plt.show()

X,Y = read()
# displayXY(X,Y)
model = fitSVC(X,Y)
Y_pred = predict(X,model)
displayRegressionLine(X,Y,Y_pred)