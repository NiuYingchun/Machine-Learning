'''逻辑回归'''
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r'D:/dataSet/')

#加载并处理数据
def splitData(file):
    data = np.loadtxt(file,delimiter=',')
    m = len(data)
    x = data[:,:-1]
    x_mean = np.mean(x,0)
    x_std = np.std(x,0)
    x = (x-x_mean) / x_std
    X = np.c_[np.ones((m,1)),x]
    y = data[:,-1].reshape(-1,1)
    return X,y
#代价函数
def costFunction(h,y,m):
    J = -(1.0/m)*(np.sum(y*np.log(h) + (1-y)*np.log(1-h)))
    return J
#逻辑函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s
#梯度下降
def grandDescent(X,y,theta,alpha,iters,m):
   # print(f'x.shape={X.shape},y.shape={y.shape}',)
    J_histories = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(X.dot(theta))
        J_histories[i] = costFunction(h,y,m)
        error = h - y
        deltaTheta = (1.0/m)*(X.T.dot(error))
       # print(deltaTheta.shape)
        theta -= alpha*deltaTheta
    return theta,J_histories
#显示分类效果
def showResult(X,y,theta,m,cost):
    for i in range(m):
        if (y[i] == 1):
            plt.plot(X[i,1],X[i,2],'rx')
        elif (y[i] == 0):
            plt.plot(X[i,1],X[i,2],'g^')

    x2 = X[:,2]
    x1 = (-theta[0]-theta[2]*x2)/theta[1]
    plt.plot(x1,x2)
    plt.show()

    plt.plot(cost)
    plt.show()


#验证分类准确率
def testModel(testX,testY,theta):
    n_samples  = len(testY)
    i = 0 #预测正确的样本个数
    for n in range(n_samples):
        h = sigmoid(testX[n].dot(theta))
        if(np.where(h>=0.5,1,0) == testY[n]):
            i+=1
    return i,i/n_samples
def main():
    x, y = splitData('ex2data1.txt')
    theta = np.ones((3, 1))
    alpha = 0.01
    iters = 5000
    m = len(x)
    theata, Cost = grandDescent(x, y, theta, alpha, iters, m)
    i,rate = testModel(x,y,theata)
    print(f'预测正确的样本个数为{i},预测正确率为{rate*100}%')
    showResult(x, y, theata, m, Cost)

if __name__ == '__main__':
   main()