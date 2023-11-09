# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task_fashion_mnist.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/31 12:13 
'''
# 基于fashion_mnist数据集，建立mlp模型，实现服饰图片十分类
# 1.实现数据加载，可视化
# 2.进行数据预处理；维度转换、归一化、输出结果格式转换
# 3.建立mlp模型，进行模型训练与预测，计算模型在训练、测试集的准确率
# 4.选取一个测试样本，预测其类别
# 5.选取测试集前10个样本，分别预测其类别
# https://github.com/zalandoresearch/fashion-mnist

# 提示：
# 模型结构：两层隐藏层（激活函数：relu），分别有392、196个神经元；输出层10类，激活函数softmax
#
# 提示2：
# 一个替代MNIST手写数字集的图像数据集， 涵盖了来自10种类别的共7万个不同服饰商品的正面图片，
# 由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度图片

#去掉AVX warning
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
#print(x_train.shape,type(x_train))#(60000, 28, 28) <class 'numpy.ndarray'>

#实现数据加载与可视化
img1=x_train[0]
# fig1=plt.figure(figsize=(5,5))
# plt.imshow(img1)
# plt.title(y_train[0])
# plt.show()

feature_size=img1.shape[0]*img1.shape[1]#行数*列数
x_train_format=x_train.reshape(x_train.shape[0],feature_size)
x_test_format=x_test.reshape(x_test.shape[0],feature_size)

x_train_normal=x_train_format/255
x_test_normal=x_test_format/255

from keras.utils import to_categorical
y_train_format=to_categorical(y_train)
y_test_format=to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Activation

fashion_mlp=Sequential()

fashion_mlp.add(Dense(units=392,activation='ReLU',input_dim=feature_size))
fashion_mlp.add(Dense(units=196,activation='ReLU'))
fashion_mlp.add(Dense(units=10,activation='softmax'))

fashion_mlp.compile(optimizer='adam',loss='categorical_crossentropy')

fashion_mlp.fit(x_train_normal,y_train_format,epochs=25)

from sklearn.metrics import accuracy_score
y_train_predict=np.argmax(fashion_mlp.predict(x_train_normal),axis=-1)
y_test_predict=np.argmax(fashion_mlp.predict(x_test_normal),axis=-1)
accuracy_train=accuracy_score(y_train,y_train_predict)
accuracy_test=accuracy_score(y_test,y_test_predict)
print('accuracy_train:',accuracy_train)#epochs=15,0.9376666666666666;epochs=50,0.9737333333333333
print('accuracy_test:',accuracy_test)#epochs=15,0.8958;epochs=50,0.8939

img2=x_test[100]
img3=x_test[252]
img4=x_test[566]
img5=x_test[123]
img6=x_test[259]
img7=x_test[159]
img8=x_test[357]
img9=x_test[153]
img10=x_test[124]

fig2=plt.figure(figsize=(10,10))

fig3=plt.subplot(331)
plt.imshow(img2)
plt.title(y_test_predict[100])
plt.ylabel(y_test[100])

fig4=plt.subplot(332)
plt.imshow(img3)
plt.title(y_test_predict[252])
plt.ylabel(y_test[252])

fig5=plt.subplot(333)
plt.imshow(img4)
plt.title(y_test_predict[566])
plt.ylabel(y_test[566])

fig6=plt.subplot(334)
plt.imshow(img5)
plt.title(y_test_predict[123])
plt.ylabel(y_test[123])

fig7=plt.subplot(335)
plt.imshow(img6)
plt.title(y_test_predict[259])
plt.ylabel(y_test[259])

fig8=plt.subplot(336)
plt.imshow(img7)
plt.title(y_test_predict[159])
plt.ylabel(y_test[159])

fig9=plt.subplot(337)
plt.imshow(img8)
plt.title(y_test_predict[357])
plt.ylabel(y_test[357])

fig10=plt.subplot(338)
plt.imshow(img9)
plt.title(y_test_predict[153])
plt.ylabel(y_test[153])

fig11=plt.subplot(339)
plt.imshow(img10)
plt.title(y_test_predict[124])
plt.ylabel(y_test[124])

plt.show()


# a = [i for i in range(1,10)]
# fig12 = plt.figure(figsize=(10,10))
# namelist=[]
# for i in y_test:
#     if i==0:
#         namelist.append('T-shirt')
#     elif i==1:
#         namelist.append('trousers')
#     elif i==2:
#         namelist.append('Pullover')
#     elif i==3:
#         namelist.append('dress')
#     elif i==4:
#         namelist.append('coat')
#     elif i==5:
#         namelist.append('sandal')
#     elif i==6:
#         namelist.append('shirt')
#     elif i==7:
#         namelist.append('sneaker')
#     elif i==8:
#         namelist.append('bag')
#     elif i==9:
#         namelist.append('Ankle boots')
#
# for i in a:
#     plt.subplot(3,3,i)
#     plt.imshow(x_test[i])
#     plt.title('predict:{}'.format(namelist[i]))
#
# plt.show()
