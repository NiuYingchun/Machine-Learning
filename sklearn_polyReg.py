from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import numpy as np
os.chdir(r'D:/dataSet/')

data = np.loadtxt('ex1data1.txt',delimiter=',')

x = data[:, 0].reshape(-1, 1)
y = data[:, -1].reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
polyX = poly.fit_transform(x)
# print(polyX)
ln = LinearRegression()
ln.fit(polyX,y)
#
xx  = np.arange(5,23).reshape(-1,1)
polyXX = poly.fit_transform(xx)
y_pred = ln.predict(polyXX)

plt.plot(xx,y_pred)
plt.scatter(x,y)
plt.show()



