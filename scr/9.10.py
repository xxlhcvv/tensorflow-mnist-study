import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets  # 导入经典数据集

# 加载 MNIST  data
(x, y), (x_test, y_test) = datasets.mnist.load_data()

print('x:', x.shape, 'y:', y.shape, 'x_test:', x_test.shape, 'y_test:', y_test.shape)
# 将加载的数据转换为 Dataset 对象
train = tf.data.Dataset.from_tensor_slices((x, y))
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 代码2-3
titanic_file = pd.read_csv('../data/titanic_file.csv')
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(titanic_file))

for feature_batch in titanic_slices.take(1):  # 采用take函数在列轴上的位置1处取值
    for key, value in feature_batch.items():   # 返回遍历的键和值
        print('{!r:20s}: {}'.format(key, value)) # 打印键与值