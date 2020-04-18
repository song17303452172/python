# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:54:54 2020

@author: 宋海沁
"""

import pandas as pd
import numpy as np

uname = ['user id','age','gender','occupation','zip code']
users = pd.read_csv('ml-100k/u.user',sep='|',names=uname)
rname = ['user id','item id','rating','timestamp']
ratings = pd.read_csv('ML-100K/u.data',sep='\t',names=rname)
users_df = users.loc[:,['user id','gender']]
ratings_df = ratings.loc[:,['user id','rating']]
rating_df = pd.merge(users_df,ratings_df)

#Way-1, groupby
'''
对数据按gender进行分组，选择分组后的rating列进行标准差运算
'''
result = rating_df.groupby('gender').rating.apply(pd.Series.std)
print(result)

#Way-1, pivot_table
result = pd.pivot_table(rating_df,values='rating',index='gender',aggfunc=pd.Series.std)
print (result)

#Way-2, groupby
'''
考虑
'''
df_temp = rating_df.groupby(['user id', 'gender']).apply(np.mean)
result = df_temp.groupby('gender').rating.apply(pd.Series.std)
print(result)

