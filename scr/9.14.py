import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# 读取数据
data = pd.read_csv('../data/line_fit_data.csv').values
# 划分训练集和测试集
x = data[: -10, 0]
y = data[: -10, 1]
x_test = data[-10: , 0]
y_test = data[-10: , 1]

# 构建Sequential网络
model_net = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),            # 使用 Input 层作为第一层，避免 Dense(input_shape=...) 的 UserWarning
    tf.keras.layers.Dense(1),                      # 全连接层
])
print(model_net.summary())
# 设置均方误差损失函数来衡量模型性能的好坏
model_net.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
# 网络训练并预测
# 通过fit方法对构建好的Sequential网络进行训练，并对测试样本的自变量进行预测
model_net.fit(x, y, verbose=1, epochs=20, validation_split=0.2)
pre = model_net.predict(x_test)
# 计算均方误差，计算样本真实值和样本预测值之间的均方误差
mse = np.mean((y_test - pre) ** 2)
print('均方误差为：', mse)