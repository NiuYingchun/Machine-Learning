import numpy as np
import os
import numpy.linalg
os.chdir(r'D:/dataSet/')

data = np.loadtxt('ex1data1.txt',delimiter=',')
m = len(data)
x = np.c_[np.ones((m,1)),data[:,0]]
y = data[:,1].reshape(-1,1)

theat = (np.linalg.inv(x.T.dot(x))).dot(x.T).dot(y)
y_pred = x.dot(theat)

result = np.c_[y,y_pred,y-y_pred]
print(result)