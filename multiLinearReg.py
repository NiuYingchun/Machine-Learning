#用单变量线性回归实现餐馆利润预测

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'D:/dataSet/')

def loadData(file):
    #加载数据
    data = np.loadtxt(file,delimiter=',')
    m = len(data)
    #print(data.shape)
    x = data[:,:-1]
    x_mean = np.mean(x,0)
    x_std = np.std(x,0)

    x = (x-x_mean) / x_std

    print(x)
    X = np.hstack((np.ones((m,1)),x))


    y = data[:,-1].reshape(-1,1)


    return X,y

def showResult(testX,testY,theta,m,cost):
    # plt.plot(X[:,1],y,'rx')
    # xx = np.arange(5,23)
    # yy = theta[0] + theta[1]*xx
    # plt.plot(xx,yy)
    # plt.show()
    y_pred = testX.dot(theta)
    plt.scatter(y_pred,testY)
    plt.show()

    plt.plot(cost)
    plt.show()
    #print(theta[0] + theta[1]*10)
#真实值   预测值  误差
# 340000    330000  10000


def costFunction(X,y,theta,m):
    #代价函数
    J = 1.0/(2*m)*(X.dot(theta) - y).T.dot(X.dot(theta) - y)
    return J

def grandescend(alpha,X,y,theta,m,iterations):
    J_histories = np.zeros(iterations)
    for i in range(iterations):
        J_histories[i] = costFunction(X,y,theta,m)
        deltaTheta = 1.0/m*(X.T.dot(X.dot(theta) - y))
        theta -= alpha*deltaTheta

    return theta,J_histories
if __name__ == '__main__':
    X,y = loadData('ex1data2.txt')
    m = len(X)
    theta = np.ones((3,1))
    theta,cost = grandescend(0.01,X,y,theta,m,1000)
    showResult(X,y,theta,m,cost)

