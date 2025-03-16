import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model = Sequential([
    Dense(units=25,activation='relu'),
    Dense(units=15,activation='relu'),
    Dense(units=10,activation='softmax'),
])



model.compile(loss=SparseCategoricalCrossentropy())  #分类别的 稀疏分类交叉熵
model.fit(X,Y,epochs=100)