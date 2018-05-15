import numpy as np
import matplotlib.pyplot as plt
#特征缩放
def featureScaling(x):
    x_mean = np.mean(x,0)
    x_std = np.std(x,0,ddof=1)
    x -=x_mean
    x /=x_std
    x = np.c_[(np.ones((m,1)),x)]
    return x
#逻辑函数(sigmod函数)
def sigmod(z):
    s = 1.0/(1+np.exp(-z))
    return s
#梯度下降算法
def gradDesc(X,y,theta,alpha,iters):
    for i in range(iters):
        # 逻辑回归模型
        h = sigmod(np.dot(X,theta))
        #建模误差
        error = h - y
        #更新theta
        theta -= alpha*(1.0/m)*(X.T.dot(error))
    return theta
def show(X,y,theta):
    for i in range(m):
        if (y[i,0] == 0):
            plt.plot(X[i,1],X[i,2],'rx')
        if (y[i,0] == 1):
            plt.plot(X[i,1],X[i,2],'b*')
    x1 = X[:,1]
    x2 = (-theta[0] - theta[1]*x1)/theta[2]#使sigmod函数为零，得到分界线
    plt.plot(x1,x2)
    plt.show()
#测试所拟合模型在测试集上的准确性
def testModel(testX,testY,theta):
    n = 0
    for i in range(m):
        h = sigmod(np.dot(testX[i,:],theta))
        if(np.where(h > 0.5 , 1 ,0) == y[i,0]):
            n += 1
    return n/m
#读取数据
data = np.loadtxt('ex2data1.txt', delimiter=',')
#获取样本个数
m = len(data)
#拿到特征
x = data[:,:-1]
X = featureScaling(x)
y = data[:,-1].reshape(-1,1)
#调用梯度下降算法，获得theta值
theta = gradDesc(X,y,[[0],[0],[0]],0.01,10000)
print(theta)
rate = testModel(X,y,theta)
print(rate)
#画模型与数据分布图
show(X,y,theta)