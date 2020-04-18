# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:26:11 2020

@author: 宋海沁
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore') 

try:
    wine = pd.read_csv('winequality-red.csv',sep=';')
except  Exception as e:
    print(e)

#删除重复数据
wine = wine.drop_duplicates()
#按葡萄酒品质进行分类统计绘制饼状图
wine['quality'].value_counts().plot(kind = 'pie', autopct = '%.2f')
plt.show()
 
#展示quality与个属性之间的相关系数
print(wine.corr().quality)
 
plt.subplot(121)
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
plt.subplot(122)
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
plt.show()
 
from sklearn.preprocessing import LabelEncoder
bins = (2, 4, 6, 8)
group_names  = ['low', 'medium', 'high']
wine['quality_lb'] = pd.cut(wine['quality'], bins = bins, labels = group_names)



print(wine.quality_lb.value_counts())



lb_quality = LabelEncoder()
wine['label'] = lb_quality.fit_transform(wine['quality_lb']) 

print(wine.label.value_counts())


wine_copy = wine.copy()
wine.drop(['quality', 'quality_lb'], axis = 1, inplace = True) 

X = wine.iloc[:,:-1]
y = wine.label


#分类任务，将数据分为测试集和训练集
from sklearn.model_selection import train_test_split   # 划分函数
#从样本中选取测试数据和训练数据，test_size设置测试集的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

#数据规范化，使用scale函数进行标准化处理
from sklearn.preprocessing import scale     
X_train = scale(X_train)
X_test = scale(X_test)

from sklearn.metrics import confusion_matrix
#随机森林函数构建分类器，n_estimators建立子数的数，子数越多模型性能越好，代码变慢
rfc = RandomForestClassifier(n_estimators = 200)
#fit()方法进行机器学习
rfc.fit(X_train, y_train)
# 基于测试集的X部分进行预测
y_pred = rfc.predict(X_test)
#混淆矩阵
print(confusion_matrix(y_test, y_pred))
 
'''数据训练，调参，另一种方式'''
param_rfc = {
            "n_estimators": [10,20,30,40,50,60,70,80,90,100,150,200],
            "criterion": ["gini", "entropy"]
            }
grid_rfc = GridSearchCV(rfc, param_rfc, iid = False, cv = 5)
grid_rfc.fit(X_train, y_train)
best_param_rfc = grid_rfc.best_params_
print(best_param_rfc)
rfc = RandomForestClassifier(n_estimators = best_param_rfc['n_estimators'], criterion = best_param_rfc['criterion'], random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
