# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Try_CNN_model.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/6 18:52 
'''
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten

model=Sequential()
model.add(Conv2D(16,(5,5),input_shape=(511,511,3),strides=(2,2),activation='ReLU'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128,(3,3),strides=(2,2),activation='ReLU'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(256,(3,3),strides=(2,2),activation='ReLU'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(512,(3,3),activation='ReLU',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='ReLU'))
model.add(Dense(units=4,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()



