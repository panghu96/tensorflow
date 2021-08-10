"""
tensorflow读取二进制文件
1.找到文件，创建文件列表
2.创建文件列表队列
3.创建文件阅读器，加载数据
4.解码，处理数据。
5.批处理
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class bin_file(object):
    """加载二进制文件
    """

    def __init__(self, file_list):
        self.file_list = file_list
        # 定义属性
        self.height = 32
        self.width = 32
        self.channels = 3
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channels
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # 2.创建文件列表队列
        file_queue = tf.train.string_input_producer(self.file_list)
        # 3.创建文件阅读器，加载数据。record_bytes 每个样本的字节大小
        reader = tf.FixedLengthRecordReader(record_bytes=self.bytes)
        # 返回k-v格式的数据，key是文件名，value是文件内容
        key, value = reader.read(file_queue)
        # 4.解码，处理数据
        label_image = tf.decode_raw(value, tf.uint8)
        # 处理数据，切分特征和标签
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])
        # 处理数据，固定形状（统一特征维度）
        image_reshape = tf.reshape(image, [self.width, self.height, self.channels])
        # 5.批处理
        label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=20, num_threads=1, capacity=20)
        return label_batch, image_batch


if __name__ == '__main__':
    # 1.找到文件，创建文件列表
    file_names = os.listdir('../data/cifar-10-binary/')
    file_list = [os.path.join('../data/cifar-10-binary/', file_name) for file_name in file_names if
                 file_name[-3:] == 'bin']
    # 实例化
    bin_tfrec = bin_file(file_list)
    label_batch, image_batch = bin_tfrec.read_and_decode()
    # 会话运行
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()
        # 启动加载数据的线程
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run([label_batch, image_batch]))
        # 等待线程结束
        coord.request_stop()
        coord.join(threads)
