# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：BP_demo2.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/7 16:22 
'''
import numpy as np


# 定义双曲函数
def tanh(x):
    return np.tanh(x)


# 双曲函数导函数
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


# 定义逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# 定义逻辑函数导函数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='logistic'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []  # 定义权重
        for i in range(1, len(layers) - 1):  # 权重初始化随机赋值
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
        # print(self.weights)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        '''
        X:输入的数据
        y:预测标记
        learning_rate:学习率
        epochs:设置算法执行次数
        '''
        X = np.atleast_2d(X)  # 转为一个m*n的矩阵
        temp = np.ones([X.shape[0], X.shape[1] + 1])  # 初始化一个m*(n+1)的矩阵  X.shape=(4,2)
        temp[:, 0:-1] = X  # 阈值的赋值
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 从0到第m-1行随机取一个数
            a = [X[i]]  # 把该行赋值给i
            # 正向更新每个神经元的输出
            for l in range(len(self.weights)):
                sum_weights = np.dot(a[l], self.weights[l])
                a.append(self.activation(sum_weights))
            error = y[i] - a[-1]  # 真实值与预测值的差
            deltas = [error * self.activation_deriv(a[-1])]  # 计算输出层的误差  对应于输出层计算式

            # 反向更新权重
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))  # 计算隐藏层误差 对应于隐藏层计算式
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)  # 权重更新计算式

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


nn = NeuralNetwork([2, 2, 1], 'tanh')
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(x, y)
for i, d in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
    print("输入值=", d, "预测值=", nn.predict(d), "真实值=", y[i])
