# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:32:43 2020

@author: 宋海沁
"""

import numpy as np
from sklearn.cluster import KMeans
list1=[88,64,96,85]
list2=[92,99,95,94]
list3=[91,87,99,95]
list4=[78,99,97,81]
list5=[88,78,98,84]
list6=[100,95,100,92]
data = np.array([list1,list2,list3,list4,list5,list6])
#fit()进行机器学习
Kmeans = KMeans(n_clusters=2).fit(data)
#根据聚类的结果分类所属的类别
pred = Kmeans.predict(data)
print(pred)