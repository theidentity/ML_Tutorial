import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Read
data = np.loadtxt("pressure")
print(data)

# Split
X = data[:,2]
Y = data[:,3]

# Reshape
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

# Fit
model = linear_model.LinearRegression()
model.fit(X,Y)

# Predict
Y_pred = model.predict(X)
print(Y_pred)

# Display
plt.plot(X,Y,'ro',color='red')
plt.plot(X,Y_pred,color='blue')
plt.show()