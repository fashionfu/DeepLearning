# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Task2-generateLetter.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/9 14:58 
'''
# 任务：基于flare文本数据，建立LSTM模型，预测序列文字
# 1.完成数据预处理，将文字序列数据转化为可用于LSTM输入的数据
# 2.查看文字数据预处理后的数据结构，并进行数据分离操作
# 3.针对字符串输入（‘flare is a teacher in ai industry.He obtained his phd in Australia.’）预测其对应
# 的后续字符
# 备注：模型结构：单层LSTM，输出有20个神经元；每次使用前20个字符预测第21个字符

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#load the data
data=open('flare').read()
data=data.replace('\n','').replace('\r','')
# print(data)

#字符去重处理
letters=list(set(data))
# print(letters)
# #['r', 'b', 'm', 'n', 'e', 'c', 'a', 's', 't', 'f', ' ', 'l', 'i', 'u', 'h', 'd', 'y', 'o', 'p', 'S', 'A', 'H', '.']
num_letters=len(letters)
# print(num_letters)#23,也就是建立的字典有23页

#建立字典（用数值进行索引，可以对应出字母的字典）
#下列创建字典过程，每次更新后的字典都不一样
#int to char
int_to_char={a:b for a,b in enumerate(letters)}#enumerate方法是将索引和字母对应起来，然后{}创建字典
# print(int_to_char)
# {0: 'r', 1: '.', 2: 'S', 3: 'A', 4: 'a', 5: 'c', 6: 'o', 7: 'i', 8: 'e',
#  9: 'y', 10: 'p', 11: 'n', 12: ' ', 13: 't', 14: 's', 15: 'u', 16: 'l', 17: 'f',
#  18: 'h', 19: 'm', 20: 'H', 21: 'b', 22: 'd'}
char_to_int={b:a for a,b in enumerate(letters)}
# print(char_to_int)
# {'l': 0, 'y': 1, 'u': 2, 'b': 3, 'p': 4, 's': 5, 'H': 6, 't': 7, 'n': 8,
#  '.': 9, 'd': 10, 'A': 11, 'S': 12, 'o': 13, 'm': 14, ' ': 15,
#  'r': 16, 'e': 17, 'i': 18, 'c': 19, 'h': 20, 'a': 21, 'f': 22}

#time_step
time_step=20#给定步长

import numpy as np
from keras.utils import to_categorical
#滑动窗口提取数据
def extract_data(data, slide):
    x = []
    y = []
    for i in range(len(data) - slide):#获取len(data)-slide组样本
        x.append([a for a in data[i:i+slide]])#一组样本
        y.append(data[i+slide])#预测值，为样本组后一位
    return x,y#循环结束后，返回数组x、y

#字符到数字的批量转化
def char_to_int_Data(x,y, char_to_int):
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[char] for char in x[i]])
        y_to_int.append([char_to_int[char] for char in y[i]])
    return x_to_int, y_to_int

#实现输入字符文章的批量处理，输入整个字符、滑动窗口大小、转化字典
def data_preprocessing(data, slide, num_letters, char_to_int):
    char_Data = extract_data(data, slide)
    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    #⭐⭐⭐⭐⭐⭐⭐⭐
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())
    Input_RESHAPED = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(0,10,size=[Input_RESHAPED.shape[0],Input_RESHAPED.shape[1],num_letters])
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i,j,:] = to_categorical(Input_RESHAPED[i,j],num_classes=num_letters)
    return new, Output

# new,Output=data_preprocessing(data,time_step,num_letters,char_to_int)
# print(new.shape,np.array(Output).shape)#(56148, 20, 23) (56148,)

#extract X and y from text data
X,y=data_preprocessing(data,time_step,num_letters,char_to_int)#遍历，将每个X转换为one-hot独热编码格式，y转换为数值
# print(np.array(X[0]).shape)#(20, 23)
# print(np.array(y).reshape(-1,1).shape)#(56148,)#(56148, 1)
# print(X.shape)#(56148, 20, 23),56148个样本
# print(y)

#Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=10)
# print(X_train.shape)#(50533, 20, 23)

y_train_category=to_categorical(y_train,num_letters)#还要给对应的字典页数
# print(y_train_category)

#set up the model
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM

model=Sequential()
model.add(LSTM(units=20,input_shape=(X_train.shape[1],X_train.shape[2]),activation='relu'))
model.add(Dense(units=num_letters,activation='softmax'))#输出层，要看字典有多少页进行输出
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#train the model
model.fit(X_train,y_train_category,batch_size=1000,epochs=10)

#make prediction based on the training data
y_train_predict=np.argmax(model.predict(X_train),axis=-1)#给出分类的结果
# print(y_train_predict)#[21  8 20 ... 17  8 21]

#transform the int to letter
# print(int_to_char[y_train_predict[1]])#每次输出y_train_predict[1]的数据都是e，代表即使每次生成的字典不一样，但是最终预测结果不会改变

y_train_predict_char=[]
y_train_predict_char.append([int_to_char[i] for i in y_train_predict])
# print(np.array(y_train_predict_char).shape)#(1, 50533)

from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train,y_train_predict)
print(accuracy_train)#1.0

y_test_predict=np.argmax(model.predict(X_test),axis=-1)
accuracy_test=accuracy_score(y_test,y_test_predict)
print(accuracy_test)#1.0
# print(y_test[0:5])
# print(y_test_predict[0:5])

new_letters='flare is a teacher in ai industry.He obtained his phd in Australia.'

X_new,y_new=data_preprocessing(new_letters,time_step,num_letters,char_to_int)
y_new_predict=np.argmax(model.predict(X_new),axis=-1)
# print(y_new_predict)
# print(y_new)

y_new_predict_char=[int_to_char[i] for i in y_new_predict]
print(y_new_predict_char)

for i in range(0,X_new.shape[0]):
    print(new_letters[i: i+20],'--predicr next new letter is--',y_new_predict_char[i])

# LSTM文本生成实战summary：
# 1.通过搭建LSTM模型，实现了基于文本序列的字符生成功能
# 2.学习了文本加载、字典生成方法
# 3.掌握了文本的数据预处理方法，并熟悉了转化数据的结构
# 4.实现了对新文本数据的字符预测