# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task2_VGG16_mlp.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/3 21:00 
'''
# 使用VGG16的结构提取图像特征，再根据特征建立特征mlp模型，实现猫狗图像识别。训练/测试数据：dataset\data_vgg:
# 1.对数据进行分离、计算测试数据预测准确率
# 2.从网站下载猫/狗图片，对其进行预测
# mlp模型一个隐藏层，10个神经元

#去掉AVX warning
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#load the data
from keras.utils.image_utils import img_to_array,load_img
img_path='1.jpg'#single image
img=load_img(img_path,target_size=(224,224))#给出图像路径和想要的尺寸，（224，224）的图像应用于VGG模型中
img=img_to_array(img)#转换成图像的数组
#print(type(img))#<class 'numpy.ndarray'>

from keras.applications.vgg16 import VGG16,preprocess_input
import numpy as np
model_vgg=VGG16(weights='imagenet',include_top=False)#不需要VGG16的全连接层，自己更改
#weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
#include_top：是否保留顶层的3个全连接网络
#去除原VGG-16中的全连接层FC与softmax后的输出，替换成mlp模型中的隐层与输出二分类，可实现猫狗预测
#需要添加维度以用于VGG模型的预测
x=np.expand_dims(img,axis=0)
x=preprocess_input(x)
#print(x.shape,type(x))#(1, 224, 224, 3) <class 'numpy.ndarray'>
#1=一张图片；224*224 3channel的图片

#特征提取
#此时的model_vgg模型已经不具备全连接层了，如果进行模型.predict后得到的将是特征数量，也就是没有进行全连接之前的层
features=model_vgg.predict(x)
#print(features.shape)#(1, 7, 7, 512)   7*7*512

#flatten将数据展开
features=features.reshape(1,7*7*512)
#print(features.shape)#(1, 25088)

#visualize the data
#进行单张图片的可视化
from matplotlib import pyplot as plt
# fig=plt.figure(figsize=(5,5))
img=load_img(img_path,target_size=(224,224))
# plt.imshow(img)
# plt.show()

#============以下已经完成了VGG-16模型的前期批处理=================
# load image and preprocess it with vgg16 structure
# --by flare
from keras.utils.image_utils import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model_vgg = VGG16(weights='imagenet', include_top=False)

# define a method to load and preprocess the image
def modelProcess(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x_vgg = model.predict(x)
    x_vgg = x_vgg.reshape(1, 25088)
    return x_vgg


# list file names of the training datasets
import os

folder = "dataset/data_vgg/cats"
dirs = os.listdir(folder)
# generate path for the images
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]

# preprocess multiple images
features1 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features1[i] = feature_i

folder = "dataset/data_vgg/dogs"
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]
features2 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features2[i] = feature_i

# label the results
print(features1.shape, features2.shape)#(300, 25088) (300, 25088) 25088个特征信息点
y1 = np.zeros(300)
y2 = np.ones(300)

# generate the training data
X = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(X.shape, y.shape)#(600, 25088) (600, 1)

#============以上已经完成了VGG-16模型的前期批处理=================

#split the training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)
print(X_train.shape,X_test.shape,X.shape)#(420, 25088) (180, 25088) (600, 25088)

#set up the mlp model
from keras.models import Sequential
from keras.layers import Dense,Activation
model=Sequential()
model.add(Dense(units=10,activation='ReLU',input_dim=25088))
#一个隐藏层输出是10；激活函数使用ReLU，最后才使用sigmoid；输入维度input_dimension是25088
model.add(Dense(units=1,activation='sigmoid'))#加入一个输出层
model.summary()#25w个参数，比之前建立的CNN模型的参数少一半多
#模型训练快，准确率比较高

#configure the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#train the model
model.fit(X_train,y_train,epochs=50)

from sklearn.metrics import accuracy_score
#✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳
#对于输出层激活函数为sigmoid的模型，其输出只有一个数，
#利用方法要将输出映射到一个标签上，应该利用np.int64(y_train_predict>0.5)进行相应的转换
# https://blog.csdn.net/qq_37654517/article/details/116327770
y_train_predict=model.predict(X_train)
y_train_predict=np.int64(y_train_predict > 0.5)
# print(y_train_predict)
#✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳
accuracy_train=accuracy_score(y_train,y_train_predict)
print(accuracy_train)#1.0

#测试集准确率预测
y_test_predict=model.predict(X_test)
y_test_predict=np.int64(y_test_predict>0.5)
accuracy_test=accuracy_score(y_test,y_test_predict)
print(accuracy_test)#0.9277777777777778
#此时的准确率远高于Task1-CNN模型，采用了更加好的VGG16模型

#进行图形可视化操作
img_path='dog_test.jpg'
img=load_img(img_path,target_size=(224,224))#给出图像路径和想要的尺寸，（224，224）的图像应用于VGG模型中
img=img_to_array(img)#转换成图像的数组
x=np.expand_dims(img,axis=0)
x=preprocess_input(x)
features=model_vgg.predict(x)
features=features.reshape(1,7*7*512)
result=model.predict(features)
print(result)#[[1.]]，此时预测正确为狗1


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
    img = load_img(img_name, target_size=(224, 224))
    img = img_to_array(img)
    x=np.expand_dims(img,axis=0)
    x=preprocess_input(x)

    x_vgg=model_vgg.predict(x)
    x_vgg=x_vgg.reshape(1,25088)
    result = model.predict(x_vgg)
    print(result[0])
    # print(result)

    img_ori = load_img(img_name, target_size=(250, 250))

    plt.subplot(3,3,i)
    plt.imshow(img_ori)
    plt.title('预测为：狗狗' if result[0] > 0.5 else '预测为：猫咪')
plt.show()

# 基于VGG16、结合mlp实现猫狗识别图像实战summary：
# 1.基于经典的VGG16结构，实现了图像识别模型的快速搭建与训练，并完成猫狗识别任务
# 2.掌握了拆分已经训练好的模型结构的方法，实现对其灵活运用
# 3.更熟练的运用mlp模型，并将其与其他模型结合，实现更复杂的任务
# 4.通过VGG16+MLP的模型，实现了在小数据集情况下的模型快速训练并获得较高的准确率























