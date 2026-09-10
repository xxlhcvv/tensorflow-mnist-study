import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets  # 导入经典数据集

# 加载 MNIST  data
(x, y), (x_test, y_test) = datasets.mnist.load_data()

print('x:', x.shape, 'y:', y.shape, 'x_test:', x_test.shape, 'y_test:', y_test.shape)