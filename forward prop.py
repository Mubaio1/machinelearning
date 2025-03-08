import numpy as np
import torch

x = np.array([200,17])

w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x)+b
a1_1 = sigmoid(z1_1)  #sigmoid: a = g(w*x+b)

w1_2 = np.array([1,2])
b1_2 = np.array([-1])
z1_2 = np.dot(w1_1,x)+b
a1_2 = sigmoid(z1_1)  #sigmoid: a = g(w*x+b)

w1_3 = np.array([1,2])
b1_3 = np.array([-1])
z1_3 = np.dot(w1_1,x)+b
a1_3 = sigmoid(z1_1)  #sigmoid: a = g(w*x+b)

a1 = np.array([a1_1,a1_2,a1_3])#为每一个神经元编码




#################实现前向传播的一般形式


def dense(a_in,W,b,g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w,a_in)+b[j]
        a_out[j] = g(z)
    return a_out


def sequential(x):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)
    a3 = dense(a2,W3,b3)
    a4 = dense(a3,W4,b4)
    f_x = a4
    return f_x