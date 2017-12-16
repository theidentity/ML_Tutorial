import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def read():
	data = np.loadtxt("pressure")
	X = data[:,2]
	Y = data[:,3]
	return X,Y

def displayXY(X,Y):
	plt.plot(X,Y,'ro')
	plt.show()

def fitRegressionLine(X,Y):
	X = X.reshape(-1,1)
	Y = Y.reshape(-1,1)
	model = linear_model.LinearRegression()
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
model = fitRegressionLine(X,Y)
Y_pred = predict(X,model)
displayRegressionLine(X,Y,Y_pred)