import numpy as np
import torch


x = np.array([[200.0,17.0]])
layer_1 = Dense(units=3,activation='sigmoid')
a1 = layer_1(x) 

layer_2 = Dense(units=1,activation='sigmoid')
a2 = layer_2(a1)

if a2 >= 0.5:
    yhat = 1
else:
    yhat = 0

model = Sequential([layer_1,layer_2])#连接两个模型  简化代码
model.compile(...)#编译模型

x = np.array([...])
y = np.array([1,0])

model.fit(x,y) #将一二层连接在一起 在数据x和y上训练

#模型建立好 后续如果输入x_new，进行预测与前向传播
model.predict(x_new)


