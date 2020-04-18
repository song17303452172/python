# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:58:31 2020

@author: 宋海沁
"""

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = [item[0] for item in iris.data]
y = [item[2] for item in iris.data]
print(x)
plt.scatter(x[:50],y[:50], color='r',marker='o')
plt.scatter(x[50:100],y[50:100], color='g',marker='o')
plt.scatter(x[100:],y[100:], color='b',marker='o')
plt.legend(loc='best')
plt.show