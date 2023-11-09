# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Learn_Task_GuiZhou.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/9 18:51 
'''
# 基于 chapter9_task_dat_train 数据，建立 rnn 模型，预测贵州茅台次日股价。
# 1、完成基本的数据加载、可视化工作；
# 2、数据预处理：将数据转化为符合 RNN 模型输入要求的数据；
# 3、建立 RNN 模型并训练模型，计算训练集、测试集模型预测 r2 分数；
# 4、可视化预测表现；
# 5、将测试数据 (chapter9_task_data_test.csv) 预测结果保存到本地 csv 文件
# 提示：
# 模型结构：单层 RNN，5 个神经元；次使用前 10 个数据预测第 11 个数据，素材参见 git

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('chapter9_task_data_train.csv')

price=data.loc[:,'close']

price_norm=price/max(price)

def extract_data(data,time_step):
    x=[]
    y=[]
    for i in range(len(data)-time_step):
        x.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    x=np.array(x)
    x=x.reshape(x.shape[0],x.shape[1],1)
    y = np.array(y)
    return x,y

time_step=10
x,y=extract_data(price_norm,time_step)

#set up the model
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
model=Sequential()
model.add(SimpleRNN(units=5,input_shape=(time_step,1),activation='ReLU'))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

model.fit(x,y,batch_size=30,epochs=200)

y_train_predict=model.predict(x)*max(price)
y=y.tolist()
y_train=[i*max(price) for i in y]

from sklearn.metrics import r2_score,explained_variance_score
R2=r2_score(y_train,y_train_predict)
evs=explained_variance_score(y_train,y_train_predict)
print(R2,evs)

fig2=plt.figure()
plt.plot(y_train,label='real price')
plt.plot(y_train_predict,label='predict price')
plt.title('close price1')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
# plt.show()

data_test=pd.read_csv('chapter9_task_data_test.csv')
price_test=data_test.loc[:,'close']
price_test_norm=price_test/max(price_test)

x_test,y_test=extract_data(price_test_norm,time_step)

y_test_predict=model.predict(x_test)
fig3=plt.figure()
plt.plot(y_test,label='real price')
plt.plot(y_test_predict,label='predict price')
plt.title('close price2')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

result_y_test=np.array(y_test).reshape(-1,1)
result_y_test_predict=y_test_predict
result=np.concatenate((result_y_test,result_y_test_predict),axis=1)
result=pd.DataFrame(result,columns=['y_test','y_test_predict'])
result.to_csv('predict_GuiZhou.csv')






