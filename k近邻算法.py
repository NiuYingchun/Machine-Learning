
import numpy as np

def createData():
    x = np.array([[1,2],[2,3],[1.1,2.2],[1.5,2.1]])
    y = np.array(['A','A','B','B'])
    return x,y
def knn(x,y,testX,k):
    m = x.shape[0]
    diff = np.tile(testX,(m,1)) - x
    squarediff = diff ** 2
    sumdiff = np.sum(squarediff,1)
    dst = sumdiff ** 0.5
    dstIndex = np.argsort(dst)
    #print(dstIndex)
    labelCount = {}
    for i in range(k):
        label = y[dstIndex[i]]
        labelCount[label] = labelCount.get(label,0) + 1
    maxCount = 0
    for k , v in labelCount.items():
        if v > maxCount:
            maxCount = v
            classes = k
    print(labelCount)
    return classes
if __name__ == '__main__':
    x,y = createData()
    testX = np.array([1.1,1.9])
    k = 3
    testY = knn(x,y,testX,k)
    print(testY)