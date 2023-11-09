# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：BP_demo.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/11/7 16:02 
'''
import numpy as np

w=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
b=[0,0.35,0.65]
l=[0,5,10]

def f1(x):
    return np.tanh(x)

def f2(x):
    return 1/(1+np.exp(-x))

def BP(w,b,l):
    h1 = f2(w[1] * l[1] + w[2] * l[2] + b[1])
    h2 = f2(w[3] * l[1] + w[4] * l[2] + b[1])
    h3 = f2(w[5] * l[1] + w[6] * l[2] + b[1])

    o1 = f2(w[7] * h1 + w[9] * h2 + w[11] * h3 + b[2])
    o2 = f2(w[8] * h1 + w[10] * h2 + w[12] * h3 + b[2])

    e=np.square(0.01-o1)/2.0+np.square(0.99-o2)/2.0

    t1=-(0.01-o1)*o1*(1-o1)
    t2 = -(0.99 - o2) * o2 * (1 - o2)

    w[7] = w[7] - 0.5 * (t1 * h1)
    w[9] = w[9] - 0.5 * (t1 * h2)
    w[11] = w[11] - 0.5 * (t1 * h3)

    w[8] = w[8] - 0.5 * (t2 * h1)
    w[10] = w[10] - 0.5 * (t2 * h2)
    w[12] = w[12] - 0.5 * (t2 * h3)

    w[1] = w[1] - 0.5 * (t1 * w[7] + t2 * w[8] * h1 * (1 - h1) * l[1])
    w[2] = w[2] - 0.5 * (t1 * w[7] + t2 * w[8] * h1 * (1 - h1) * l[2])
    w[3] = w[3] - 0.5 * (t1 * w[9] + t2 * w[10] * h2 * (1 - h2) * l[1])
    w[4] = w[4] - 0.5 * (t1 * w[9] + t2 * w[10] * h2 * (1 - h2) * l[2])
    w[5] = w[5] - 0.5 * (t1 * w[11] + t2 * w[12] * h3 * (1 - h3) * l[1])
    w[6] = w[6] - 0.5 * (t1 * w[11] + t2 * w[12] * h3 * (1 - h3) * l[2])

    return o1,o2,w,e

for i in range(101):
    o1,o2,w,e=BP(w,b,l)
    print('第{}次迭代：预测值：（{},{}）,总误差：{}，权重系数：{}'.format(i,o1,o2,e,w))
