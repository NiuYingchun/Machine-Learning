#多项式回归（成功）
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
def get_data( ):
    X_parameter = pd.DataFrame([0.5, 0.8, 1.6, 1.0, 0.6, 2.8, 1.2, 0.9, 0.4, 2.3, 2.2, 4.0, 2.6, 2.8, 4.2])
    Y_parameter = pd.DataFrame([1.1, 1.9, 2.2, 2.3, 1.6, 4.9, 2.8, 2.1, 1.4, 3.4, 3.4, 5.8, 5.0, 5.4, 6.0])
    data = pd.concat([X_parameter,Y_parameter],axis=1)
    return data
data=get_data()
data.columns = ['licheng','price']
data = data.sort_values(['licheng','price'])
X=data['licheng'].as_matrix().reshape(-1,1)
Y=data['price'].as_matrix().reshape(-1,1)
poly = PolynomialFeatures(2)
polyX = poly.fit_transform(X)
regr = LinearRegression()
#用数据生成模型
model = regr.fit(polyX, Y)
deg = model.coef_.shape[1]-1
pred_y = regr.predict(polyX)
plt.scatter(X, Y, color='blue')
plt.plot(polyX[:,1], pred_y,'g-', label='degree ' + str(deg) + ' fit')
plt.legend(loc='upper left')
plt.show()