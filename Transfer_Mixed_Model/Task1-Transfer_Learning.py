# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task1-Transfer_Learning.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/15 9:09 
'''
# 任务：基于transfer_data.csv数据，建立mlp模型，再实现模型迁移学习
# 1.实现x对y的预测，可视化结果
# 2.基于新数据transfer_data2.csv，对前模型进行二次训练，对比模型训练次数少的情况下的表现
# 备注：模型结构：mlp，两个隐藏层，每层50个神经元，激活函数relu，输出层激活函数linear，迭代次数：100次

#load the data
import pandas as pd
import numpy as np
import tensorflow.python.keras.models
from matplotlib import pyplot as plt

data=pd.read_csv('transfer_data.csv')
x=data.loc[:,'x']
y=data.loc[:,'y']

# fig1=plt.figure(figsize=(7,5))
# plt.scatter(x,y)
# plt.title('y vs x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

x=np.array(x).reshape(-1,1)
# print(x,x.shape)#(101, 1)

from keras.models import Sequential
from keras.layers import Dense

model1=Sequential()
model1.add(Dense(units=50,input_dim=1,activation='relu'))
model1.add(Dense(units=50,activation='relu'))
model1.add(Dense(units=1,activation='linear'))

model1.compile(optimizer='adam',loss='mean_squared_error')
model1.summary()

model1.fit(x,y,epochs=400)

y_predict=model1.predict(x)
# print(y_predict)

from sklearn.metrics import r2_score
R2=r2_score(y,y_predict)
# print(R2)#0.998900520641369,epochs=400

# fig2=plt.figure(figsize=(7,5))
# raw=plt.scatter(x,y)
# prediction=plt.plot(x,y_predict,'r')
# plt.title('y vs x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

model1.save('model1.h5')
import tensorflow.keras.models
model2=tensorflow.keras.models.load_model('model1.h5')

#如果使用joblib，其新版本不能保存深度学习模型

data2=pd.read_csv('transfer_data2.csv')
x2=data2.loc[:,'x2']
y2=data2.loc[:,'y2']

x2=np.array(x2).reshape(-1,1)

y2_predict=model2.predict(x2)#此处仍针对之前的初始数据进行了模型的拟合

fig3=plt.figure(figsize=(7,5))
fig4=plt.subplot(121)
plt.scatter(x,y,label='data1')
plt.scatter(x2,y2,label='data2')
plt.plot(x2,y2_predict,'r',label='predict2')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.show()

#transfer learning，考虑使用迁移学习方法，将新数据给到模型进行拟合
model2.fit(x2,y2,epochs=400)
y2_predict_2=model2.predict(x2)
fig5=plt.subplot(122)
plt.scatter(x,y,label='data1')
plt.scatter(x2,y2,label='data2')
plt.plot(x2,y2_predict_2,'r',label='predict2')
plt.title('y vs x(transfer learning)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()#与上方图像相比，预测曲线拟合的位置给到了新的数据点

# 基于新数据的迁移学习实战summary：
# 1.通过使用新数据，实现了模型的2次训练，达到了较好的预测效果
# 2.建立mlp模型实现了非线性分布数据的回归预测
# 3.掌握了模型存储与加载的方法
# 4.通过迁移学习，可减少模型训练迭代次数，大幅缩短训练时间
