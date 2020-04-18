# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:11:31 2020

@author: 宋海沁
"""

#导入scipy库，库中已经有实现的kmeans模块，直接使用，
#根据六个人的分数分为学霸或者学渣两类
import numpy as np
from scipy.cluster.vq import vq,kmeans,whiten 
list1=[88,64,96,85]
list2=[92,99,95,94]
list3=[91,87,99,95]
list4=[78,99,97,81]
list5=[88,78,98,84]
list6=[100,95,100,92]
data = np.array([list1,list2,list3,list4,list5,list6])
#数据归一化处理
whiten = whiten(data)
print(whiten)
#使用kmeans聚类，第一个参数为数据，第二个参数是k类，得到的结果是二维的，
#所以加一个下划线表示不取第二个值，第一个值为得到的聚类中心，第二个值为损失
centroids,_= kmeans(whiten,2)
#使用vq函数根据聚类中心将数据进行分类，输出的结果为二维，第一个结果为分类的标签，第二个结果不需要。
result,_=vq(whiten,centroids)
print(result)
