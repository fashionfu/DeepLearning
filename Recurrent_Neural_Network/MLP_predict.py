# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：MLP_predict.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/9 21:16 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import pandas as pd
import numpy as np
data_train=pd.read_csv('chapter9_task_data_train.csv')
# data_train.head()

price_train=data_train.loc[:,'close']
price_train_norm=price_train/max(price_train)

def extract_data(data,time_step):
    x=[]
    y=[]
    for i in range(len(data)-time_step):
        x.append([a for a in data[i:i+time_step]])
        y.append([data[i+time_step]])
    x=np.array(x)
    y=np.array(y)
    return x,y

time_step=10
x_train_norm,y_train_norm=extract_data(price_train_norm,time_step)
# print(x_train_norm.shape,y_train_norm.shape)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=x_train_norm.shape[1],units=10,activation='relu'))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train_norm,y_train_norm,epochs=200)

y_predict_train=model.predict(x_train_norm)*max(price_train)
y_train=y_train_norm*max(price_train)

from sklearn.metrics import r2_score
r2_score(y_train,y_predict_train)

from matplotlib import pyplot as plt
fig1=plt.figure(figsize=(8,5))
plt.plot(y_train,label='real_price_train')
plt.plot(y_predict_train,label='predict_price_train')
plt.legend()
plt.show()

data_test=pd.read_csv('chapter9_task_data_test.csv')
price_test=data_test.loc[:,'close']
price_test_norm=price_test/max(price_train)
x_test_norm,y_test_norm=extract_data(price_test_norm,time_step)
y_test=y_test_norm*max(price_train)

y_predict_test=model.predict(x_test_norm)*max(price_train)
y_test=y_test_norm*max(price_train)
r2_score(y_test,y_predict_test)

fig2=plt.figure(figsize=(8,5))
plt.plot(y_test,label='real_price_test')
plt.plot(y_predict_test,label='predict_price_test')
plt.title('mlp_test:r2=0.968')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
