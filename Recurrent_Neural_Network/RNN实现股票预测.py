# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：RNN实现股票预测.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/10 9:22 
'''
# -*- coding: utf-8 -*-
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间


def get_stock_data():
    # 没有进行归一化
    # 获取原始股票数据
    maotai = pd.read_csv('')  # 读取股票文件

    # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，   2:3 是提取[2:3)列，前闭后开,即提取出C列开盘价
    training_set = maotai.iloc[0:2426 - 300, 2:3].values
    # # 后300天的开盘价作为测试集
    test_set = maotai.iloc[2426 - 300:, 2:3].values

    # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
    # training_set_scaled 归一化的训练集
    training_set_scaled = sc.fit_transform(training_set)
    # 归一化的测试集
    test_set_scaled = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

    x_train = []
    y_train = []
    # 训练数据
    # 利用for循环 连续60天作为输入特征 第61天作为输出特征
    for i in range(60, len(training_set_scaled)):
        x_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # 对训练集进行打乱
    np.random.seed(7)
    np.random.shuffle(x_train)
    np.random.seed(7)
    np.random.shuffle(y_train)

    # 将训练集由list格式变为array格式
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
    # 送入样本数: x_train.shape[0]即2066组数据；
    # 循环核时间展开步数: 输入60个开盘价，
    # 每个时间步输入特征个数:  每个时间步送入的特征是某一天的开盘价，只有1个数据，故为1
    x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

    # 测试数据
    x_test = []
    y_test = []
    for i in range(60, len(test_set_scaled)):
        x_test.append(test_set_scaled[i - 60:i, 0])
        y_test.append(test_set_scaled[i, 0])
    # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # 使x_test符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
    x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))
    return (x_train, y_train), (x_test, y_test)


def load_local_model(model_path):
    if os.path.exists(model_path + ''):
        print('-------------load the model-----------------')
        print(datetime.datetime.now())
        local_model = tf.keras.models.load_model(model_path)
    else:
        local_model = tf.keras.Sequential([
            SimpleRNN(80, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(100),
            Dropout(0.2),
            Dense(1)
        ])
        local_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                            loss='mean_squared_error')  # 损失函数用均方误差
        # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
    return local_model


def show_train_line(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def stock_predict(model, x_test, y_test):
    # 测试集输入模型进行预测
    predicted_stock_price = model.predict(x_test)
    print(predicted_stock_price)
    # 对预测数据还原---从（0，1）反归一化到原始范围
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # 对真实数据还原---从（0，1）反归一化到原始范围
    real_stock_price = sc.inverse_transform(np.reshape(y_test, (y_test.shape[0], 1)))
    # 画出真实数据和预测数据的对比曲线
    plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
    plt.title('MaoTai Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MaoTai Stock Price')
    plt.legend()
    plt.show()

    ##########evaluate##############
    # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
    mse = mean_squared_error(predicted_stock_price, real_stock_price)
    # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
    rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
    # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
    mae = mean_absolute_error(predicted_stock_price, real_stock_price)
    print('均方误差: %.6f' % mse)
    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_stock_data()
    model_path = ""
    model = load_local_model(model_path)
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test),
                        validation_freq=1)
    show_train_line(history)
    model.summary()
    model.save(model_path, save_format="tf")
    stock_predict(model, x_test, y_test)
    print(datetime.datetime.now())

