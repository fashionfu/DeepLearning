# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：error_test.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/3 18:49 
'''
#✳✳✳✳✳✳✳✳✳✳以下为task1-cnn-cat_dog的测试文档✳✳✳✳✳✳✳✳✳✳
# #去掉AVX warning
# import os
# import random
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
#
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
#
# # Step 9: ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale = 1./255)
#
#
#
# # Step 10: Load the training Set
# training_set = train_datagen.flow_from_directory('./Dataset/training_set',
#                                                  target_size = (50, 50),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# #set up CNN model
#
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
#
# # Step 2: Initialising the CNN
# model = Sequential()
# # Step 3: Convolution
# model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
# # Step 4: Pooling
# model.add(MaxPooling2D(pool_size = (2, 2)))
# # Step 5: Second convolutional layer
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# # Step 6: Flattening
# model.add(Flatten())
# # Step 7: Full connection
# model.add(Dense(units = 128, activation = 'relu'))
# model.add(Dense(units = 1, activation = 'sigmoid'))
# # Step 8: Compiling the CNN
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#
#
# # Step 11: Classifier Training
# model.fit_generator(training_set,epochs = 1)
#
# print(model.evaluate_generator(training_set))
#
# test_set = train_datagen.flow_from_directory('./Dataset/test_set',
#                                                  target_size = (50, 50),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# print(model.evaluate_generator(test_set))
#
# # Step 5: load the image you want to test
# # coding:utf-8
# import matplotlib as mlp
# font2 = {'family' : 'SimHei',
# 'weight' : 'normal',
# 'size'   : 20,
# }
# mlp.rcParams['font.family'] = 'SimHei'
# mlp.rcParams['axes.unicode_minus'] = False
# from matplotlib import pyplot as plt
# from matplotlib.image import imread
# from keras.utils.image_utils import load_img,img_to_array
# from keras.models import load_model
# #from cv2 import load_img
# a = [i for i in range(1,10)]
# fig = plt.figure(figsize=(10,10))
# for i in a:
#     img_name = str(i)+'.jpg'
#     img_ori = load_img(img_name, target_size=(50, 50))
#     img = img_to_array(img_ori)
#     img = img.astype('float32')/255
#     img = img.reshape(1,50,50,3)
#     result=model.predict(img)
#     img_ori = load_img(img_name, target_size=(250, 250))
#     plt.subplot(3,3,i)
#     plt.imshow(img_ori)
#     plt.title('预测为：狗狗' if result[0] > 0.5 else '预测为：猫咪')
# plt.show()
#✳✳✳✳✳✳✳✳✳✳以上为task1-cnn-cat_dog的测试文档✳✳✳✳✳✳✳✳✳✳



