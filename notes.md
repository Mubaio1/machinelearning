TensorFlow   pytorch

一     神经网络 TensorFlow(neutral network)

给出一个向量  在神经网络中进行推理
每个神经元都要进行一次向量计算  推理下一个a[i]

sigmoid函数

eg：咖啡
x = np.array([[200.0,17.0]])
layer_1 = Dense(units=3,activation='sigmoid')  // 3个单元  使用sigmoid激活 Dense是所谓的全连接层
a1 = layer_1(x)  //将layer_1用作第一个隐藏层



Tensor出现在代码中  本质是Tensor团队指定的易于运算和存储的向量   张量（Tensor）与矩阵类似  是矩阵的一般或者表示方式

 eg:
 a1 = layer_1(x)
 tf.Tensor([[0.2 0.7 0.3]],shape=(1,3),dtype=float32)

将Tensor转化为Numpy格式返回   则执行a1.numpy()  即可将数组以numpy的形式返回




二   构建神经网络

使用几个函数来对模型进行指定参数与编译训练等  而后进行拟合

