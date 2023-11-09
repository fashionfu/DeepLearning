# # -*- coding: UTF-8 -*-
# '''
# @Project ：PycharmDemo
# @File    ：demo_fashion.py
# @IDE     ：PyCharm
# @Author  ：10208
# @Date    ：2022/10/31 15:03
# '''
# #!/usr/bin/env python
# # coding: utf-8
#
# #去掉AVX warning
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
#
# from keras.datasets import fashion_mnist
# (X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
# print(type(X_train),X_train.shape)
# img1 = X_train[0]
# from matplotlib import pyplot as plt
# fig1 = plt.figure(figsize=(3,3))
# plt.imshow(img1)
# plt.title(y_train[0])
# plt.show()
# feature_size = img1.shape[0]*img1.shape[1]
# X_train_format = X_train.reshape(X_train.shape[0],feature_size)
# X_test_format = X_test.reshape(X_test.shape[0],feature_size)
# print(X_train_format.shape)
# X_train_normal = X_train_format/255
# X_test_normal = X_test_format/255
# from tensorflow.keras.utils import to_categorical
# y_train_format = to_categorical(y_train)
# y_test_format = to_categorical(y_test)
# print(y_train_format[0])
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# mlp = Sequential()
# mlp.add(Dense(units=392,activation=‘relu’,input_dim=784))
# mlp.add(Dense(units=192,activation=‘relu’))
# mlp.add(Dense(units=10,activation=‘softmax’))
# mlp.summary()
# mlp.compile(loss=‘categorical_crossentropy’,optimizer=‘adam’,metrics=[‘categorical_accuracy’])
# mlp.fit(X_train_normal,y_train_format,epochs=10,batch_size=1)
# import numpy as np
# y_train_predict = mlp.predict(X_train_normal)
# y_train_predict = np.round(y_train_predict).astype(int)
# from sklearn.metrics import accuracy_score
# accuracy_train = accuracy_score(y_train_format,y_train_predict)
# print(accuracy_train)
# y_test_predict = mlp.predict(X_test_normal)
# y_test_predict = np.round(y_test_predict).astype(int)
# accuracy_test = accuracy_score(y_test_format,y_test_predict)
# print(accuracy_test)
# import matplotlib as mlp
# font2 = {‘family’ : ‘SimHei’,
# ‘weight’ : ‘normal’,
# ‘size’ : 20,
# }
# mlp.rcParams[‘font.family’] = 'SimHei’
# mlp.rcParams[‘axes.unicode_minus’] = False
# a = [i for i in range(1,10)]
# fig4 = plt.figure(figsize=(10,10))
# namelist=[]
# for i in y_test:
#     if i==0:
#         namelist.append('T恤/上衣')
#     elif i==1:
#         namelist.append('裤子')
#     elif i==2:
#         namelist.append(‘套头衫’)
#     elif i==3:
#         namelist.append(‘连衣裙’)
#     elif i==4:
#         namelist.append(‘外套’)
#     elif i==5:
#         namelist.append(‘凉鞋’)
#     elif i==6:
#         namelist.append(‘衬衫’)
#     elif i==7:
#         namelist.append(‘运动鞋’)
#     elif i==8:
#         namelist.append(‘包’)
#     elif i==9:
#         namelist.append(‘踝靴’)
# for i in a:
#     plt.subplot(3,3,i)
#     plt.tight_layout()
#     plt.imshow(X_test[i])
#     plt.title(‘predict:{}’.format(namelist[i]),font2)
#     plt.xticks([])
#     plt.yticks([])
#
