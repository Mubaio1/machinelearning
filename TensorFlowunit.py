import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
    Dense(units=25,activation='sigmoid'),
    Dense(units=15,activation='sigmoid'),
    Dense(units=1,activation='sigmoid'),
])

model.compile(loss=BinaryCrossentropy())
model.fit(X,Y,epochs=100)#X和Y为数据集 epochs轮次  梯度下降步数

#fit自动实现了前向传播、计算损失、方向传播、迭代重复的过程
