# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:02:37 2020

@author: 宋海沁
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    df = pd.read_csv('数据\StudentPerformance.csv')
except Exception as e:
    print(e)

'''
属性介绍：
Gender: 性别
Nationality: 国籍
PlaceofBirth：出生地
StageID：学校级别（小学，中学，高中）
GradeID：年级 (G01 - G12)
SectionID: 班级
Topic：学科科目
Semester: 学期 （春学期，秋学期）
Relation: 孩子家庭教育负责人（父亲，母亲）
RaisedHands: 学生该学期上课举手的次数
VisitedResources: 学生浏览在线课件的次数
AnnoucementsView: 学生浏览学校公告的次数
Discussion: 学生参与课堂讨论的次数
ParentAnsweringSurvey: 家长是否填写了关于学校的问卷调查 （是，否）
ParentSchoolSatisfaction: 家长对于学校的满意度 （好，不好）
StudentAbsenceDays: 学生缺勤天数 （大于7天，低于7)

结果(Response Variable)介绍：
Class: 根据学生最后的学术评测分数，学生会被分为3个等级
Low-Level: 分数区间在0-60
Middle-Level:分数区间在70-89
High-Level:分数区间在90-100
'''
#判断是否有空值
print(df.isnull().sum())
#查看所有特征的统计信息
print(df.describe(include='all'))

#产看特征分类
print('gender(性别)',df['gender'].unique())
print('NationalITy(国籍)',df['NationalITy'].unique())
print('PlaceofBirth(出生地)',df['PlaceofBirth'].unique())
print('StageID(学校级别)',df['StageID'].unique())
print('GradeID(年级)',df['GradeID'].unique())
print('SectionID(班级)',df['SectionID'].unique())
print('Topic(学科科目)',df['Topic'].unique())
print('Semester(学期)',df['Semester'].unique())
print('Relation(孩子家庭教育负责人)',df['Relation'].unique())

#查看数据集的结果是否平衡
plt.style.use('ggplot')
sns.countplot(x='Class',order=['L','M','H'],data=df)

'''
数据可视化分析：
'''

#gender分布情况
plt.figure(figsize=(10,5))

f,[ax1,ax2]=plt.subplots(1,2)
sns.countplot(x='gender',data=df,ax=ax1)
sns.countplot(x='gender',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.title('性别-成绩',fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#NationalITy(国籍) 分布情况
f,[ax1,ax2]=plt.subplots(2,1,figsize=(12,8))
sns.countplot(x='NationalITy',data=df,ax=ax1)
sns.countplot(x='NationalITy',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.title('国籍-成绩',fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#学校级别分布
f,[ax1,ax2]=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='StageID',data=df,ax=ax1)
sns.countplot(x='StageID',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.title('学校-成绩',fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#Topic(学科科目)
f,[ax1,ax2]=plt.subplots(2,1,figsize=(10,5))
sns.countplot(x='Topic',data=df,ax=ax1)
sns.countplot(x='Topic',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.title('学科科目-成绩',fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#了解班级和成绩的相关性
f,[ax1,ax2]=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='SectionID',data=df,ax=ax1)
sns.countplot(x='SectionID',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.title('班级-成绩',fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#以上的特征和Class结果没有什么直接的关系，在后续的处理中可以考虑移除特征

'''
数字型特征可视化
'''
'''
raisedhands: 学生该学期上课举手的次数
VisITedResources: 学生浏览在线课件的次数
AnnouncementsView: 学生浏览学校公告的次数
Discussion: 学生参与课堂讨论的次数
'''
#课堂表现与成绩的关系
ClassroomPerformance=['raisedhands','VisITedResources',\
                       'AnnouncementsView','Discussion']
fig = plt.figure(figsize=(10,8))
for i in range(4):
    ax=plt.subplot(2,2,i+1)
    sns.barplot(x='Class',y=df[ClassroomPerformance[i]],data=df)
    plt.title('{0}-成绩'.format(ClassroomPerformance[i]),fontproperties="SimHei",fontsize=12)
plt.tight_layout()

#说明这四个方面和成绩有很大关系，我们通过相关性矩阵来检验一下
'''
相关矩阵
'''
corr=df[['VisITedResources','raisedhands','AnnouncementsView','Discussion']].corr()
fig = plt.figure(figsize=(16,12))
ax=sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
ax.set_ylim([4, 0])



#ParentAnsweringSurvey: 家长是否填写了关于学校的问卷调查 （是，否）和成绩的相关度
f,[ax1,ax2]=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='ParentAnsweringSurvey',data=df,ax=ax1)
sns.countplot(x='ParentAnsweringSurvey',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.tight_layout()
#ParentSchoolSatisfaction: 家长对于学校的满意度 （好，不好）和成绩的相关度
f,[ax1,ax2]=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='ParentschoolSatisfaction',data=df,ax=ax1)
sns.countplot(x='ParentschoolSatisfaction',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.tight_layout()
#StudentAbsenceDays  学生缺勤天数 （大于7天，低于7天）
f,[ax1,ax2]=plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='ParentAnsweringSurvey',data=df,ax=ax1)
sns.countplot(x='ParentAnsweringSurvey',hue='Class',hue_order=['L','M','H'],data=df,ax=ax2)
plt.tight_layout()


'''
模型训练1
'''
df['handsAndVistResourse'] = df['raisedhands'] * df['VisITedResources'] 
df['handsAndAnnouncementsView'] = df['raisedhands'] * df['AnnouncementsView']   

x=df.drop(['gender','SectionID','StageID','GradeID','Topic','NationalITy',\
           'PlaceofBirth','Semester','Class'],axis=1)
x = pd.get_dummies(x)
y=df['Class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
# Fit and Predict # 训练模型并检测准确率
Logit = LogisticRegression()
Logit.fit(x_train, y_train)
Predict=Logit.predict(x_test)
Score=accuracy_score(y_test,Predict)
print(Score)








