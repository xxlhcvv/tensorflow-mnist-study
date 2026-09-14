import tensorflow as tf

# 加载MNIST手写数字数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("训练集形状：",x_train.shape)
print("测试集形状：",x_test.shape)