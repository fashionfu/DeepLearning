# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：vgg-批处理.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/3 17:08 
'''

#去掉AVX warning
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

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
print(features1.shape, features2.shape)
y1 = np.zeros(300)
y2 = np.ones(300)

# generate the training data
X = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(X.shape, y.shape)