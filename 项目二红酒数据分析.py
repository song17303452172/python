# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:43:27 2020

@author: 宋海沁
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


color = sns.color_palette()
pd.set_option('precision',3) 

try:
    df = pd.read_csv('数据/winequality-red.csv',sep = ';')
except Exception as e:
    print(e)

'''
单变量分析
'''
  
#更改画图风格
plt.style.use('ggplot')

#绘制箱型图
colnm = df.columns.tolist() #pandas 中 DataFramt 改变 列的顺序
fig = plt.figure(figsize=(12,8))

for i in range(12):
    plt.subplot(2,6,i+1)
    sns.boxplot(df[colnm[i]],orient='v',width=0.5,color=color[1])  #orient:“v” | “h”，可选,控制绘图的方向（垂直或水平）
    plt.ylabel(colnm[i],fontsize=12)
plt.tight_layout()

#绘制直方图

colnm = df.columns.tolist()
fig = plt.figure(figsize=(12,10))

for i in range(12):
    plt.subplot(3,4,i+1)
    df[colnm[i]].hist(bins=100,color=color[0])
    plt.xlabel(colnm[i],fontsize=12)
    plt.ylabel('frequency')
plt.tight_layout()  # 自适应xlabel、ylabel的位置


acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
               'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']

plt.figure(figsize = (10, 4))

for i in range(6):
    ax = plt.subplot(2,3,i+1)
    #clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
    v = np.log10(np.clip(df[acidityFeat[i]].values, a_min = 0.001, a_max = None))
    plt.hist(v, bins = 50, color = color[0])
    plt.xlabel('log(' + acidityFeat[i] + ')',fontsize = 12)
    plt.ylabel('Frequency')
plt.tight_layout()


plt.figure(figsize=(6,3))

bins = 10**(np.linspace(-2, 2))
plt.hist(df['fixed acidity'], bins = bins, edgecolor = 'k', label = 'Fixed Acidity')
plt.hist(df['volatile acidity'], bins = bins, edgecolor = 'k', label = 'Volatile Acidity')
plt.hist(df['citric acid'], bins = bins, edgecolor = 'k', alpha = 0.7, label = 'Citric Acid')
plt.xscale('log')# 显示x轴坐标时，显示对数
plt.xlabel('Acid Concentration (g/dm^3)')
plt.ylabel('Frequency')
plt.title('Histogram of Acid Concentration')
plt.legend()
plt.tight_layout()

df['total acid']=df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
plt.figure(figsize = (8,3))

plt.subplot(121)
plt.hist(df['total acid'], bins = 50, color = color[0])
plt.xlabel('total acid')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(np.log(df['total acid']), bins = 50 , color = color[0])
plt.xlabel('log(total acid)')
plt.ylabel('Frequency')
plt.tight_layout()

'''
甜度（sweetness）：Residual sugar 与酒的甜度相关通常用来区别各种红酒，干红（<=4 g/L),
半干（4-12 g/L）,半甜（12-45 g/L），和甜（>45 g/L)。 这个数据中，主要为干红，没有甜葡萄酒。
'''

#pandas.cut用来把一组数据分割成离散的区间。按bins指定的区间进行分类，并且区间名字为labels
df['sweetness'] = pd.cut(df['residual sugar'], bins = [0, 4, 12, 45], 
                      labels=["dry", "medium dry", "semi-sweet"])
plt.figure(figsize = (5,3))
#value_counts()相当于分类求和
df['sweetness'].value_counts().plot(kind = 'bar', color = color[0])
plt.xticks(rotation=0)
plt.xlabel('sweetness', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.tight_layout()

'''
双变量分析
'''

#红酒品质与理化特征的关系
colnm = df.columns.tolist()[:11] + ['total acid']
plt.figure(figsize = (10, 8))

for i in range(12):
    plt.subplot(4,3,i+1)
    sns.boxplot(x ='quality', y = colnm[i], data = df, color = color[1], width = 0.6)
    plt.ylabel(colnm[i],fontsize = 12)
plt.tight_layout()


'''
#有问题，不能进行输出，显示输出不为整数
sns.set_style("dark")

plt.figure(figsize = (10,8))
colnm = df.columns.tolist()[:11] + ['total acid', 'quality']
mcorr = df[colnm].corr()
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.palplot(sns.diverging_palette(240, 10, n=9))
cmap = sns.diverging_palette(220, 10,as_cmap=True)
#cmap = sns.diverging_palette(as_cmap=True)
try:    
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True)
except Exception as e:
    print(e)
print("\nFigure 8: Pairwise Correlation Plot")    
'''


#密度与酒精浓度的关系
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)

plt.figure(figsize = (6,4))
#sns.regplot可视化线性回归关系和implot的做那个用基本一致推荐使用regplot
sns.regplot(x='density', y = 'alcohol', data = df, scatter_kws = {'s':10}, color = color[0])
plt.xlim(0.989, 1.005)
plt.ylim(7,16)


#酸性物质含量和PH
acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide', 
                   'sulphates', 'total acid']

plt.figure(figsize = (10,6))
for i in range(5):
    plt.subplot(2,3,i+1)
    sns.regplot(x='pH', y = acidity_related[i], data = df, scatter_kws = {'s':10}, color = color[1])
plt.tight_layout()


#多变量关系
'''
与品质相关性最高的三个特征是酒精浓度，挥发性酸度，和柠檬酸。下面图中显示的酒精浓度，挥发性酸和品质的关系。
酒精浓度，挥发性酸和品质
对于好酒（7，8）以及差酒（3，4），关系很明显。但是对于中等酒（5，6），酒精浓度的挥发性酸度有很大程度的交叉。
'''

plt.style.use('ggplot')


'''
hue:条件在第三个变量上并绘制不同颜色的水平
 size是高aspect是宽占高的比例。更改构面的高度和纵横比
 col_wrap多行显示子图
 scatter_kws定义回归拟合
 fit_reg是否启用回归拟合
'''
sns.lmplot(x = 'alcohol', y = 'volatile acidity', hue = 'quality', 
           data = df, fit_reg = False, scatter_kws={'s':10}, size = 5)

sns.lmplot(x = 'alcohol', y = 'volatile acidity', col='quality', hue = 'quality', 
           data = df,fit_reg = True, size = 3,  aspect = 0.9, col_wrap=3,
           scatter_kws={'s':50})


#pH，非挥发性酸，和柠檬酸
'''
pH和非挥发性的酸以及柠檬酸有相关性。整体趋势也很合理，即浓度越高，pH越低。
'''
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)

plt.figure(figsize=(6,5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], c=df['pH'], vmin=2.6, vmax=4, s=15, cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation = 0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4,18)
plt.ylim(0,1)
plt.tight_layout()


'''
总结：
整体而言，红酒的品质主要与酒精浓度，挥发性酸，和柠檬酸有关。
对于品质优于7，或者劣于4的酒，直观上是线性可分的。但是品质为5，6的酒很难线性区分。
'''
