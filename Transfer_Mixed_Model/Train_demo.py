# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Train_demo.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/16 16:58 
'''
import numpy as np
img_path=[0,1,2,3,4,5,6,7,8,9]
features_train=np.zeros([len(img_path),10])
for i in range(len(img_path)):
    features_train[i]=i*10

print(features_train)



