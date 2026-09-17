import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets  # 导入经典数据集
import os
import matplotlib.pyplot as plt

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

titanic_batches = tf.data.experimental.make_csv_dataset(
    file_pattern='../data/titanic_file.csv',
    batch_size=4,
    column_names=None,
    column_defaults=None,
    label_name='survived',
    select_columns=None,
    field_delim=',',
    use_quote_delim=True,
    na_value='',
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
    encoding='utf-8'
)

titanic_batches = tf.data.experimental.make_csv_dataset('../data/titanic_file.csv', batch_size=4, label_name='survived')
for feature_batch, label_batch in titanic_batches.take(1):
    print('survived: {}'.format(label_batch))
    print('features:')
    for key, value in feature_batch.items():
        print('{!r:20s}: {}'.format(key, value))

dataset = tf.data.TFRecordDataset(filenames = ['../data/fsns.tfrec'])
print(dataset)

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy()) # 对TFRecord进行解码
print(parsed.features.feature['image/text'])              # 输出检查

cowper = tf.data.TextLineDataset('../data/cowper.txt')
for line in cowper.take(5):
    print(line.numpy())

import random
import pathlib

data_path = pathlib.Path('../data/flower_photos')
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths] # 所有图片路径的列表
random.shuffle(all_image_paths) # 打散数据

image_count = len(all_image_paths)
print('数据大小：', image_count)
# 查看5张图片
print('5张图片', all_image_paths[:5])

# 提取分类名
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
print('分类名', label_names)

# 创建标签
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print('标签', label_to_index)

# 将图片与标签对应
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
for image, label in zip(all_image_paths[:5], all_image_labels[:5]):
    print(image, ' ---> ', label)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 代码2-10：将加载后的图片转化为Dataset对象
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 新的预处理函数
def parse_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label

# 代码2-12 调用预处理函数
images_ds = ds.map(parse_image)

# 取一张图片测试绘图
for image, label in images_ds.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()

