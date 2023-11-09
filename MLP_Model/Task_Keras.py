# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task_Keras.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/27 11:06 
'''
#建立MLP实现非线性二分类
#任务：基于data.csv数据，建立mlp模型，计算其在测试数据上的准确率，可视化模型预测结果
#test_size=0.33,random_state=10
#模型结构：一层隐藏层，有20个神经元

#去掉AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#load the data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
#简单来说，to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。
#其表现为将原有的类别向量转换为独热编码的形式

#define the X and y
data=pd.read_csv('data.csv')
X=data.drop(['y'],axis=1)
y=data.loc[:,'y']
#使用one-hot编码，将类别变量转换为机器学习易于利用的一种形式的过程
y=to_categorical(y)
#one-hot编码：https://blog.csdn.net/weixin_41857483/article/details/111396939

#visualize the data
# fig1=plt.figure(figsize=(5,5))
# #在对y进行one hot编码后，使用np.argmax(y,axis=-1)可以重新找出返回最大概率的索引
# #np.argmax()的介绍：https://www.jianshu.com/p/3649fef6bceb
# #axis的相关介绍：https://blog.csdn.net/sky_kkk/article/details/79725646
# passed=plt.scatter(X.loc[:,'x1'][np.argmax(y,axis=-1)==1],X.loc[:,'x2'][np.argmax(y,axis=-1)==1])
# failed=plt.scatter(X.loc[:,'x1'][np.argmax(y,axis=-1)==0],X.loc[:,'x2'][np.argmax(y,axis=-1)==0])
# plt.legend((passed,failed),('passed','failed'))
# plt.title('raw data')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)
#print(X_train.shape,X_test.shape,X.shape)#(275, 2) (136, 2) (411, 2)

#set up the model
from keras.models import Sequential
#序列模型：https://blog.csdn.net/baidu_41867252/article/details/89065739
from keras.layers import Dense,Activation

mlp=Sequential()
mlp.add(Dense(units=20,input_dim=2,activation='sigmoid'))
mlp.add(Dense(units=2,activation='sigmoid'))

# mlp.summary()

#compile the model
mlp.compile(optimizer='adam',loss='binary_crossentropy')

#train the model
mlp.fit(X_train,y_train,epochs=3000)

# https://mp.weixin.qq.com/s/vb5rUril7ZiQN5hkzfeD3A
# 干货｜keras中model.predict_classes报错处理方法

#make prediction and calculate the accuracy
y_train_predict=np.argmax(mlp.predict(X_train),axis=-1)
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(np.argmax(y_train,axis=-1),y_train_predict)
print('training data:',accuracy_train)#PS：每次的准确率都不一样
#建议在管道中运行几次训练和测试作业，并计算运行中评估指标的均值和标准差，可以用来可靠的预测是否使用了模型

#make the prediction based on the test data
y_test_predict=np.argmax(mlp.predict(X_test),axis=-1)
accuracy_test=accuracy_score(np.argmax(y_test,axis=-1),y_test_predict)
print('testing data',accuracy_test)

#generate new data for plot
#有关np.meshgrip()的介绍：https://blog.csdn.net/lllxxq141592654/article/details/81532855
#用于快速生成坐标矩阵X,Y
xx,yy=np.meshgrid(np.arange(0,1,0.01),np.arange(0,1,0.01))
x_range=np.c_[xx.ravel(),yy.ravel()]
y_range_predict=np.argmax(mlp.predict(x_range),axis=-1)
#print(y_range_predict)#[1 1 1 ... 1 1 1]
#print(type(y_range_predict))#<class 'numpy.ndarray'>
# 如果下行不进行reshape操作，后续会在pd.Series中提示无效索引
y_range_predict=np.array(y_range_predict).reshape(-1,1)
# print(y_range_predict)
# [[1]
#  [1]
#  [1]
#  ...
#  [1]
#  [1]
#  [1]]
#print(type(y_range_predict))#<class 'numpy.ndarray'>


#format the output
y_range_predict_form=pd.Series(i[0] for i in y_range_predict)
#print(type(y_range_predict_form))#<class 'pandas.core.series.Series'>

fig2=plt.figure(figsize=(5,5))
passed_mlp=plt.scatter(x_range[:,0][y_range_predict_form==1],x_range[:,1][y_range_predict_form==1])
failed_mlp=plt.scatter(x_range[:,0][y_range_predict_form==0],x_range[:,1][y_range_predict_form==0])

passed=plt.scatter(X.loc[:,'x1'][np.argmax(y,axis=-1)==1],X.loc[:,'x2'][np.argmax(y,axis=-1)==1])
failed=plt.scatter(X.loc[:,'x1'][np.argmax(y,axis=-1)==0],X.loc[:,'x2'][np.argmax(y,axis=-1)==0])

plt.legend((passed,failed,passed_mlp,failed_mlp),('passed','failed','passed_mlp','failed_mlp'))
plt.title('prediction results')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#summary:
#1.通过mlp模型，可以在不增加特征项的情况下实现非线性二分类任务
#2.掌握mlp模型的建立、配置与训练方法，并实现基于新数据的预测
#3.熟悉mlp分类的预测数据格式，并实现格式转换