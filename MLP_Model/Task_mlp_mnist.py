# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task_mlp_mnist.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/31 10:07 
'''
# 基于mnist数据集，建立mlp模型，实现0-9数据的十分类task：
# 1.实现mnist数据载入，可视化图形数字
# 2.完成数据预处理：图像数据维度转换与归一化、输出结果格式转换
# 3.计算模型在预测数据集的准确率
# 4.模型结构：两层隐藏层，每层有392个神经元

#去掉AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#load the mnist data
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#print(type(X_train),X_train.shape)#<class 'numpy.ndarray'> (60000, 28, 28)
#print(type(X_test),X_test.shape)#<class 'numpy.ndarray'> (10000, 28, 28)

#visualize the data of mnist
img1=X_train[0]
#print(img1.shape)#(28, 28)
#要将其转换成1行784列的数组进行模型拟合,肯定要把每个像素块的数据分别读出来才行，不是说一整个图像进行拟合
# fig1=plt.figure(figsize=(3,3))
# plt.imshow(img1)#可以把一个矩阵array直接画图画出来
# plt.title(y_train[0])#y_train是对应于X_train的标签
# plt.show()

#format the input data
feature_size=img1.shape[0]*img1.shape[1]#行数×列数
#print(feature_size)#784
X_train_format=X_train.reshape(X_train.shape[0],feature_size)
#print(X_train_format.shape)#(60000, 784),60000个样本，784列：等价于原先28*28的数组转换为784列的向量
X_test_format=X_test.reshape(X_test.shape[0],feature_size)
#print(X_test_format.shape)#(10000, 784)

#normalize the input data，归一化处理数据
X_train_normal=X_train_format/255#变成0-1的数字
X_test_normal=X_test_format/255

#format the output data(labels)，对输出结果进行类型转换
from keras.utils import to_categorical
y_train_format=to_categorical(y_train)
y_test_format=to_categorical(y_test)
#print(y_train_format[0])#[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

#set up the model
from keras.models import Sequential
from keras.layers import Dense,Activation

mlp=Sequential()
mlp.add(Dense(units=392,activation='sigmoid',input_dim=feature_size))
mlp.add(Dense(units=392,activation='sigmoid'))
mlp.add(Dense(units=10,activation='softmax'))

mlp.summary()#结构展示

#configure the model
mlp.compile(optimizer='adam',loss='categorical_crossentropy')#多分类模型中计算熵（损失函数）
#train the model
mlp.fit(X_train_normal,y_train_format,epochs=5)

#evaluate the model
y_train_predict=np.argmax(mlp.predict(X_train_normal),axis=-1)
#print(y_train_predict)#[5 0 4 ... 5 6 8]
y_test_predict=np.argmax(mlp.predict(X_test_normal),axis=-1)

from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train,y_train_predict)
print(accuracy_train)#0.99455,epochs=10
accuracy_test=accuracy_score(y_test,y_test_predict)
print(accuracy_test)#0.977,epochs=10

img2=X_test[100]
img3=X_test[101]
img4=X_test[102]
img5=X_test[103]

fig=plt.figure(figsize=(10,10))

fig2=plt.subplot(221)
plt.imshow(img2)#可以把一个矩阵array直接画图画出来
plt.title(y_test_predict[100])#y_train是对应于X_train的标签

fig3=plt.subplot(222)
plt.imshow(img3)#可以把一个矩阵array直接画图画出来
plt.title(y_test_predict[101])#y_train是对应于X_train的标签

fig4=plt.subplot(223)
plt.imshow(img4)#可以把一个矩阵array直接画图画出来
plt.title(y_test_predict[102])#y_train是对应于X_train的标签

fig5=plt.subplot(224)
plt.imshow(img5)#可以把一个矩阵array直接画图画出来
plt.title(y_test_predict[103])#y_train是对应于X_train的标签

plt.show()

# 图像数字多分类实战summary：
# 1.通过mlp模型，实现了基于图像数据的数字自动识别分类
# 2.完成了数字的图像化处理与可视化
# 3.对mlp模型的输入、输出数据格式有了更深的认识，完成了数据预处理与格式转换
# 4.建立了结构更为复杂的mlp模型
# 5.mnist数据集地址：http://yann.lecun.com/exdb/mnist/


# print(X_train_format[0])
# [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255
#  247 127   0   0   0   0   0   0   0   0   0   0   0   0  30  36  94 154
#  170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0   0   0
#    0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82
#   82  56  39   0   0   0   0   0   0   0   0   0   0   0   0  18 219 253
#  253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  35 241
#  225 160 108   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
#  253 207   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253
#  253 201  78   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0  18 171 219 253 253 253 253 195
#   80   9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#   55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0 136 253 253 253 212 135 132  16
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0]
# print(X_train_normal[0])
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.01176471 0.07058824 0.07058824 0.07058824
#  0.49411765 0.53333333 0.68627451 0.10196078 0.65098039 1.
#  0.96862745 0.49803922 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.11764706 0.14117647 0.36862745 0.60392157
#  0.66666667 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686
#  0.88235294 0.6745098  0.99215686 0.94901961 0.76470588 0.25098039
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.19215686
#  0.93333333 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686
#  0.99215686 0.99215686 0.99215686 0.98431373 0.36470588 0.32156863
#  0.32156863 0.21960784 0.15294118 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.07058824 0.85882353 0.99215686
#  0.99215686 0.99215686 0.99215686 0.99215686 0.77647059 0.71372549
#  0.96862745 0.94509804 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.31372549 0.61176471 0.41960784 0.99215686
#  0.99215686 0.80392157 0.04313725 0.         0.16862745 0.60392157
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.05490196 0.00392157 0.60392157 0.99215686 0.35294118
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.54509804 0.99215686 0.74509804 0.00784314 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.04313725
#  0.74509804 0.99215686 0.2745098  0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.1372549  0.94509804
#  0.88235294 0.62745098 0.42352941 0.00392157 0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.31764706 0.94117647 0.99215686
#  0.99215686 0.46666667 0.09803922 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.17647059 0.72941176 0.99215686 0.99215686
#  0.58823529 0.10588235 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.0627451  0.36470588 0.98823529 0.99215686 0.73333333
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.97647059 0.99215686 0.97647059 0.25098039 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.18039216 0.50980392 0.71764706 0.99215686
#  0.99215686 0.81176471 0.00784314 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.15294118 0.58039216
#  0.89803922 0.99215686 0.99215686 0.99215686 0.98039216 0.71372549
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.09411765 0.44705882 0.86666667 0.99215686 0.99215686 0.99215686
#  0.99215686 0.78823529 0.30588235 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.09019608 0.25882353 0.83529412 0.99215686
#  0.99215686 0.99215686 0.99215686 0.77647059 0.31764706 0.00784314
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.07058824 0.67058824
#  0.85882353 0.99215686 0.99215686 0.99215686 0.99215686 0.76470588
#  0.31372549 0.03529412 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.21568627 0.6745098  0.88627451 0.99215686 0.99215686 0.99215686
#  0.99215686 0.95686275 0.52156863 0.04313725 0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.53333333 0.99215686
#  0.99215686 0.99215686 0.83137255 0.52941176 0.51764706 0.0627451
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.        ]




