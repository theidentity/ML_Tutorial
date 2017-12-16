import numpy as np
import matplotlib.pyplot as plt


# Reference: https://byjus.com/linear-regression-formula

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
	xsum = np.sum(X)
	ysum = np.sum(Y)
	x2sum = np.sum(X**2)
	xysum = np.sum(X*Y)
	n = X.shape[0]

	m = float(n*xysum-xsum*ysum)/(n*x2sum-xsum**2)
	b = float(ysum-m*xsum)/n
	model = m,b
	
	return model

def predict(X,model):
	X = X.reshape(-1,1)
	m,b = model
	Y_pred = m*X + b
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