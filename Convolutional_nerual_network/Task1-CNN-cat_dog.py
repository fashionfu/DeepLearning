# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task1-CNN-cat_dog.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/3 10:00 
'''
# 基于dataset\training_set数据，根据提供的结构，建立CNN模型，识别图片中的猫/狗，计算预测准确率
# 1.识别图片中的猫/狗，计算dataset\training_set测试数据预测准确率
# 2.从网站下载猫/狗图片，对其进行预测

#去掉AVX warning
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#load the data
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)#缩放处理，使其进行归一化，模型训练时更高效
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#         rotation_range=40,#随机旋转角度数范围
#         width_shift_range=0.2,#随机宽度偏移量
#         height_shift_range=0.2,#随机高度偏移量
#         rescale=1./255,#所有数据集将乘以该数值
#         shear_range=0.2,
#         zoom_range=0.2,#随机缩放的范围 -> [1-n,1+n]
#         horizontal_flip=True,#是否随机水平翻转
#         fill_mode='nearest')
#
# data_generator = datagen.flow_from_directory('./datas/train', target_size=(224,224), batch_size=32)
# 该函数可以增强图片数据，需要fit函数来对指定的数据进行增强，这里要求是四维数据（图片张数，图片长度，图片宽度，灰度），先reshape为四维数据然后调用fit函数

training_set=train_datagen.flow_from_directory('./dataset/training_set',target_size=(50,50),batch_size=32,class_mode='binary')
#batch_size是我们希望每次训练时希望从图库中取得多少张图片

#print(training_set.class_indices)#{'cats': 0, 'dogs': 1}

#set up the cnn model
#====================建立模型==========================
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
#MaxPool2D和MaxPooling2D是同一个池化层
model=Sequential()
#卷积层
model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='ReLU'))
#池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#卷积层
model.add(Conv2D(32,(3,3),activation='ReLU'))
#池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#flattening layer
model.add(Flatten())
#FC layer
model.add(Dense(units=128,activation='ReLU'))
model.add(Dense(units=1,activation='sigmoid'))
#=================完成主模型结构配置========================

#configure the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#加入metrics是为了在训练时有一个评估的方式，训练时可以看到

model.summary()

# json_str=model.to_json()
# print(json_str)

#train the model
model.fit_generator(training_set,epochs=25)#是因为之前数据是采用数据增强image_generator产生

#accuracy on the training data
accuracy_train=model.evaluate_generator(training_set)
print(accuracy_train)#[0.0003043838369194418, 1.0]

#accuracy on the testing data
test_set=train_datagen.flow_from_directory('./dataset/test_set',target_size=(50,50),batch_size=32,class_mode='binary')
accuracy_test=model.evaluate_generator(test_set)
print(accuracy_test)#[1.6699533462524414, 0.7730000019073486]
#前一个数据是损失函数的值，第二个数据是准确率

#load the single image
from keras.utils.image_utils import load_img,img_to_array
pic_dog='dog_test.jpg'
pic_dog=load_img(pic_dog,target_size=(50,50))
pic_dog=img_to_array(pic_dog)
pic_dog=pic_dog/255#归一化处理
pic_dog=pic_dog.reshape(1,50,50,3)#转换成可以进行模型预测的图片格式
result=model.predict(pic_dog)
print('狗[1]:预测为狗狗'if result[0]>0.5 else '狗[1]:预测为猫咪')#狗[1]:[[0.9921557]]

pic_cat='cat_test.jpg'
pic_cat=load_img(pic_cat,target_size=(50,50))
pic_cat=img_to_array(pic_cat)
pic_cat=pic_cat/255#归一化处理
pic_cat=pic_cat.reshape(1,50,50,3)#转换成可以进行模型预测的图片格式
result1=model.predict(pic_cat)
print('猫[0]:预测为狗狗'if result[0]>0.5 else '猫[0]:预测为猫咪')#猫[0]: [[0.96211606]]

pic_cat_2='cat_test2.jpg'
pic_cat_2=load_img(pic_cat_2,target_size=(50,50))
pic_cat_2=img_to_array(pic_cat_2)
pic_cat_2=pic_cat_2/255#归一化处理
pic_cat_2=pic_cat_2.reshape(1,50,50,3)#转换成可以进行模型预测的图片格式
result2=model.predict(pic_cat_2)
print('猫[0]:预测为狗狗'if result2[0]>0.5 else '猫[0]:预测为猫咪' )#猫[0]: [[0.07561655]]

pic_dog_2='dog2.jpg'
pic_dog_2=load_img(pic_dog_2,target_size=(50,50))
pic_dog_2=img_to_array(pic_dog_2)
pic_dog_2=pic_dog_2/255#归一化处理
pic_dog_2=pic_dog_2.reshape(1,50,50,3)#转换成可以进行模型预测的图片格式
result3=model.predict(pic_dog_2)
print('狗[1]:预测为狗狗'if result3[0]>0.5 else '狗[1]:预测为猫咪' )#狗[1]: [[0.22307931]]

pic_dog_3='dog3.jpg'
pic_dog_3=load_img(pic_dog_3,target_size=(50,50))
pic_dog_3=img_to_array(pic_dog_3)
pic_dog_3=pic_dog_3/255#归一化处理
pic_dog_3=pic_dog_3.reshape(1,50,50,3)#转换成可以进行模型预测的图片格式
result4=model.predict(pic_dog_3)
print('狗[1]:预测为狗狗'if result4[0]>0.5 else '狗[1]:预测为猫咪')#狗[1]: [[0.96726376]]

#❌❌❌❌❌❌❌❌❌以下为旧版本且存在错误的代码❌❌❌❌❌❌❌❌❌❌❌❌
# #make prediction on multiple images
# import matplotlib as mlp
# font2 = {'family' : 'SimHei',
# 'weight' : 'normal',
# 'size'   : 20,
# }
# mlp.rcParams['font.family'] = 'SimHei'
# mlp.rcParams['axes.unicode_minus'] = False
# from matplotlib import pyplot as plt
# from matplotlib.image import imread
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
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
#     result = model.predict_classes(img)
#     img_ori = load_img(img_name, target_size=(250, 250))
#     plt.subplot(3,3,i)
#     plt.imshow(img_ori)
#     plt.title('预测为：狗狗' if result[0][0] == 1 else '预测为：猫咪')
# plt.show()
#❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌

import matplotlib as mlp
font2 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 20,
}
mlp.rcParams['font.family'] = 'SimHei'
mlp.rcParams['axes.unicode_minus'] = False
from matplotlib import pyplot as plt
from matplotlib.image import imread
from keras.utils.image_utils import load_img,img_to_array
from keras.models import load_model
#from cv2 import load_img
a = [i for i in range(1,10)]
fig = plt.figure(figsize=(10,10))
for i in a:
    img_name = str(i)+'.jpg'
    img_ori = load_img(img_name, target_size=(50, 50))
    img = img_to_array(img_ori)
    img = img.astype('float32')/255
    img = img.reshape(1,50,50,3)
    result = model.predict(img)
    print(result[0])
    # print(result)
    img_ori = load_img(img_name, target_size=(250, 250))
    plt.subplot(3,3,i)
    plt.imshow(img_ori)
    plt.title('预测为：狗狗' if result[0] > 0.5 else '预测为：猫咪')
plt.show()

# CNN实现猫狗识别实战summary：
# 1.通过搭建CNN模型，实现了对复杂图像的自动识别分类
# 2.掌握了图像数据的批量加载与图像增强方法
# 3.更熟练的掌握了keras的sequence结构，并嵌入卷积、池化层
# 4.实现了对网络图片的分类识别
# 5.图像预处理参考资料：http://keras.io/preprocessing/image/

# 修改了imagedatagenerator中的参数，添加了rotation、horizon等，并修改了迭代次数为75次
# 最终，训练集准确率为：0.9445000290870667，测试集准确率为：0.8029999732971191
# 由此可见，进行多次迭代和对图像进行旋转平移等操作，有机会提升测试集的准确率

# 再跑了一次模型(epochs=75)，训练集达到95%的准确率，但测试集仅有78%了

#预测出来的全是0，存在严重问题

#✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳
#续上，预测出来全是0，是因为判断条件为result[0]==1时给定为狗狗，而当前版本进行model.predict(img)时，给出的值是
#一个0-1之间的数值，经验就是可以把每个result打印出来，再根据这个值将判断条件进行修改
#=======================================因此=====================================================
#当我把plt.title('预测为：狗狗' if result[0] > 0.5 else '预测为：猫咪')，修改成此判断条件时，可以进行正常预测
#同时，建议不熟悉此段注释时，重新跑跑程序或看看print出来的result，一定能加深印象
#✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳
