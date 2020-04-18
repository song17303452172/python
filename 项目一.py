# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:22:51 2020

@author: 宋海沁
"""

from sklearn import datasets
import pandas as pd
import numpy as np
#选取数据
boston = datasets.load_boston()
x = pd.DataFrame(boston.data,columns=boston.feature_names)
y = pd.DataFrame(boston.target,columns=['MEdv'])

#查看相关度
import matplotlib.pyplot as plt
plt.scatter(x['RM'],y,color='blue')
plt.show()
plt.scatter(x['LSTAT'],y,color='red')
plt.show()

#调用机器学习api接口，选择T检验法，检验特征
import statsmodels.api as sm
x_add1=sm.add_constant(x)
model=sm.OLS(y,x_add1).fit()  #sm.OLS()为普通最小二乘回归模型，fit()用于拟合 

#输出结果，查看P值是否大于0.5，如果有元素大于0.5，则进行删除元素，删除后进行从新检验
print(model.summary())
#删除后进行从新检验
x.drop('INDUS',axis=1,inplace=True)
x.drop('AGE',axis=1,inplace=True)
x_add1=sm.add_constant(x) #加入常数项
model=sm.OLS(y,x_add1).fit()
print(model.summary())
#P值(p>|t|)全部不大于0.5,则完成，表中coef即为回归系数。可使用print(model.params)进行输出
print(model.params)

#生成测试数据,刚刚删除了两个即为11个加上const常数，总共12个。
x_test= np.array([1,0.006,18.0,0.0,0.52,6.6,4.87,1.0,290.0,15.2,396.2,5])
print(model.predict(x_test)) #输出预测结果



'''
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
#定义一个线性回归对象
clf = linear_model.LinearRegression()
x= np.array([2,3,4,5,6,7]).reshape(-1,1)
y = np.array([6,10,14,14.5,21,18.8])
# #fit()方法进行机器学习
clf.fit(x,y)
#训练模型
b,a=clf.coef_,clf.intercept_
print(b, a)
X = [[4]]
#用模型预测
print(clf.predict(X))
plt.scatter(x,y)
plt.plot(x,a+b*x,color='red')
plt.show()
'''
