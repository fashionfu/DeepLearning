# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：task1-NewStockPrice.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/8 18:28 
'''
# 任务：基于zgpa_train.csv数据，建立RNN模型，预测股价：
# 1.完成数据预处理，将序列数据转化为可用于RNN输入的数据
# 2.对新数据zgpa_test.csv进行预测，可视化结果
# 3.存储预测结果，并观察局部预测结果
# 备注：模型结构：单层RNN，输出有5个神经元；每次使用前8个数据预测第9个数据

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import pandas as pd
import numpy as np
data=pd.read_csv('zgpa_train.csv')
#print(data.head())

price=data.loc[:,'close']
# print(price.head())

# print(data[0:8])#取了0-7的数据出来

#归一化处理
#对模型进行归一化处理，帮助模型更好的进行收敛
price_norm=price/max(price)#0-1之间的一个值
#个人理解：此处对输入的元素进行归一化处理，是为了更好的使数据达到tanh（x）函数的效果
#也就是防止梯度消失和梯度爆炸现象的产生（simple RNN model），也就是每次进行normalization，让数据恢复到（-1，1）这个合适的区间中
#07：06  https://www.bilibili.com/video/BV1fp4y1t7Xb?vd_source=9fd5445ca85300af5a2a7ad84fffe91c
# print(price_norm)

from matplotlib import pyplot as plt
# fig1=plt.figure(figsize=(8,5))
# plt.plot(price)
# plt.title('close price')
# plt.xlabel('time')#仅使用0-700之间的数字表示时间
# plt.ylabel('price')
# plt.show()

#define X and y
#define method to extract X and y
def extract_data(data,time_step):
    X=[]
    y=[]
    #0,1,2···9：10个样本；time_step=8时，0,1,···7为第一组样本；1，2，···8为第二组样本，因为此时9后面没有10需要预测，所以共两组样本
    for i in range(len(data)-time_step):#比如10个数据，10-8=2组样本
        X.append([a for a in data[i:i+time_step]])#X的最后一个数据是i+time_step-1,X是一整组数据样本
        y.append(data[i+time_step])
    X=np.array(X)
    # print(X)
    #要将X转换为RNN模型可以识别的数据，进行输入：input_shape=(samples,time_steps,features)
    X=X.reshape(X.shape[0],X.shape[1],1)#数据序列，维度为1；而文本数据为one-hot格式，维数是feature
    return X,y

time_step = 8
X,y=extract_data(price_norm,time_step)
y=np.array(y)#这一步不知道为什么老师没进行np.array数组的转换就可以进行模型的训练了，如果不转换格式的话会报错
# print(X)
# print(X.shape)#(723, 8, 1),723个样本、8为步长、1是数据维度
# print(X[0,:,:])
# print(y[0:10])

#set up the model
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
model=Sequential()
#add RNN layer
model.add(SimpleRNN(units=5,input_shape=(time_step,1),activation='ReLU'))
#add output layer
model.add(Dense(units=1,activation='linear'))#不是进行分类，不需要softmax的激活函数了，使用线性模型即可
#configure the model
model.compile(optimizer='adam',loss='mean_squared_error')#使用线性模型中的距离平方差，均方差进行loss损失函数的评估

model.summary()

#有的小伙伴训练一次以后发现预测出来的结果不理想，很可能是模型进行初始化的时候选取的随机系数不合适，导致梯度下降搜索时遇到了局部极小值
#解决办法：尝试再次建立模型并训练
#多层感知机结构在进行模型求解时，会给定一组随机的初始化权重系数，这种情况是正常的。通常我们可以观察损失函数是否在变小来发现模型求解是否正常

#train the model
model.fit(X,y,batch_size=30,epochs=200)

#make prediction based on the training data
y_train_predict=model.predict(X)*max(price)#将之前归一化的数据转换回来
y=y.tolist()#又将前面进行np.array数组转换回了列表
y_train=[i*max(price) for i in y]
# print(y_train)
# print(y_train_predict)
#print(y_train.shape)#AttributeError: 'list' object has no attribute 'shape'
from sklearn.metrics import explained_variance_score
evs=explained_variance_score(y_train,y_train_predict)
print(evs)#0.9947384911221171

fig2=plt.figure(figsize=(8,5))
plt.plot(y_train,label='real price')
plt.plot(y_train_predict,label='predict price')
plt.title('close price zgpa_train')
plt.xlabel('time')#仅使用0-700之间的数字表示时间
plt.ylabel('price')
plt.legend()
# plt.show()

data_test=pd.read_csv('zgpa_test.csv')
price_test=data_test.loc[:,'close']
price_test_norm=price_test/max(price)
#extract X_test and y_test
X_test_norm,y_test_norm=extract_data(price_test_norm,time_step)
y_test_norm=np.array(y_test_norm)
# print(X_test_norm.shape)#(174, 8, 1)
# print(y_test_norm.shape)#(174,)

#make prediction based on the test data
y_test_predict=model.predict(X_test_norm)*max(price)#同样要进行归一化的复原
y_test=y_test_norm*max(price)#此时，在109行中已经进行了np.array数组的转换，可以直接乘上max(price)，
#而不需要像87行那样操作，那样演示是因为老师进行操作时，一直将y使用做列表，而现在的RNN模型进行fit拟合时需要用np.array数组

fig3=plt.figure(figsize=(8,5))
plt.plot(y_test,label='real price test')
plt.plot(y_test_predict,label='predict price test')
plt.title('close price zgpa_test')
plt.xlabel('time')#仅使用0-700之间的数字表示时间
plt.ylabel('price')
plt.legend()
plt.show()

result_y_test=np.array(y_test).reshape(-1,1)
result_y_test_predict=y_test_predict
#print(result_y_test.shape,result_y_test_predict.shape)#(174, 1) (174, 1)
result=np.concatenate((result_y_test,result_y_test_predict),axis=1)#进行列合并时，axis=1
# print(result.shape)#(174, 2),记得上方要将两个需要连接的数组用括号括起来，这样就转换为174行2列的数组了
result=pd.DataFrame(result,columns=['real_price_test','predict_price_test'])
result.to_csv('zgpa_predict_test1.csv')#将测试数据集对应的股价和预测结果都存储了起来

# RNN预测股价实战summary:
# 1.通过搭建RNN模型，实现了基于历史数据对次日股价的预测
# 2.熟悉了RNN模型的数据格式结构
# 3.掌握了数字序列的数据预处理方式
# 4.实现了预测数据存储，通过可视化局部细节了解了RNN用于股价预测的局限性：信息延迟