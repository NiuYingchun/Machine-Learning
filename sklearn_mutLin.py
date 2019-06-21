from sklearn.linear_model import LinearRegression
import os
import numpy as np

os.chdir(r'D:/dataSet/')
trainData = np.loadtxt('ccpp_train.txt',delimiter=',')
testData = np.loadtxt('ccpp_test.txt',delimiter=',')
trainX = trainData[:,:-1]
trainY = trainData[:,-1].reshape(-1,1)
testX = testData[:,:-1]
testY = testData[:,-1].reshape(-1,1)

ln = LinearRegression()
ln.fit(trainX,trainY)
y_pred = ln.predict(testX)
print('预测值   真实值    误差')
result = np.c_[y_pred,testY,y_pred-testY]
print(result)