from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import numpy as np
os.chdir(r'D:/dataSet/')

def loadData(file):
    data = np.loadtxt(file,delimiter=',')
    x = data[:,0].reshape(-1,1)
    y = data[:,-1].reshape(-1,1)
    return x,y
def fit(linearRegression,x,y):
    linearRegression.fit(x,y)
    theta = np.c_[linearRegression.coef_,linearRegression.intercept_]
    return theta
def predict(linearRegression,testX,testY):
    y_pred = linearRegression.predict(testX)
    return y_pred
def showResult(x,y,theta):
    xx = np.arange(4,24)
    yy = theta[0,1] + theta[0,0]*xx
    plt.plot(xx,yy)
    plt.scatter(x,y)
    plt.show()
def main():
    linearRegression = LinearRegression()
    x,y = loadData('ex1data1.txt')
    theta = fit(linearRegression,x,y)
    y_pred = predict(linearRegression,x,y)
    result = np.c_[y,y_pred,y-y_pred]
    #print(result)
    # print(theta.shape)
    # showResult(x,y,theta)

if __name__ == '__main__':
    main()

#print(result)