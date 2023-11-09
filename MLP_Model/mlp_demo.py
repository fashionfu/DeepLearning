# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：mlp_demo.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/27 20:01 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#load the data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

#define the X and y
data=pd.read_csv('data.csv')
x=data.drop(['y'],axis=1)
y=data.loc[:,'y']

#分离数据
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
y=to_categorical(y)#对原始结果标签做one-hot处理
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=10)
#print(x.shape,y.shape,x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#build the mlp model
from keras.models import Sequential
from keras.layers import Dense,Activation

mlp=Sequential()
mlp.add(Dense(units=20,input_dim=2,activation='sigmoid'))
mlp.add(Dense(units=2,activation='sigmoid'))#输出单位数对应的结果标签数
mlp.summary()

#compile the model
mlp.compile(optimizer='adam',loss='binary_crossentropy')

#train the model
mlp.fit(x_train,y_train,epochs=3000)

#make a prediction of training data
y_train_predict=np.argmax(mlp.predict(x_train),axis=-1)
#print(y_train_predict)

#calculate the training accuracy
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(np.argmax(y_train,axis=-1),y_train_predict)
print('training accuracy:',accuracy_train)
#算得结果为0.9672727272727273

#make a prediction of test data
y_test_predict=np.argmax(mlp.predict(x_test),axis=-1)
#print(y_test_predict)

#calculate the test accuracy
accuracy_test=accuracy_score(np.argmax(y_test,axis=-1),y_test_predict)
#print('testing accuracy:',accuracy_test)
#算得结果为0.9705882352941176

#generate new data for plot
xx,yy=np.meshgrid(np.arange(0,1,0.01),np.arange(0,1,0.01))
x_range=np.c_[xx.ravel(),yy.ravel()]
y_range_predict=np.argmax(mlp.predict(x_range),axis=-1)
#print(type(y_range_predict),y_range_predict)

y_range_predict=np.array(y_range_predict)
y_range_predict=y_range_predict.reshape(-1,1)
#print(type(y_range_predict),y_range_predict.shape,y_range_predict)

y_range_predict_form=pd.Series(i[0] for i in y_range_predict)
#print(y_range_predict_form)

#visualize the predict result
fig2=plt.figure(figsize=(6,6))

pass_mlp=plt.scatter(x_range[:,0][y_range_predict_form==1],x_range[:,1][y_range_predict_form==1])
fail_mlp=plt.scatter(x_range[:,0][y_range_predict_form==0],x_range[:,1][y_range_predict_form==0])

y = np.argmax(y,axis=-1)
passed=plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1])
failed=plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0])

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((passed,failed,pass_mlp,fail_mlp),('passed','failed','pass_mlp','fail_mlp'))
plt.title('predict result')
plt.show()