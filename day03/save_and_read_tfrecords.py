"""
将文件保存成tfrecords格式，并加载
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class tfrecords_file(object):
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
        """
        加载二进制文件
        :return: label_batch 批处理的标签, image_batch 批处理的特征
        """
        # 创建文件队列
        file_queue = tf.train.string_input_producer(self.file_list)
        # 创建阅读器，加载数据。record_bytes 每个样本的字节数
        reader = tf.FixedLengthRecordReader(record_bytes=self.bytes)
        key, value = reader.read(file_queue)
        # 解码，处理数据
        label_image = tf.decode_raw(value, tf.uint8)
        # 切分标签和特征
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])
        # 固定形状（统一特征维度）
        image_reshape = tf.reshape(image, [self.height, self.width, self.channels])
        # 批处理
        label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=20, num_threads=1, capacity=20)
        return label_batch, image_batch

    def save_to_tfrecords(self, label_batch, image_batch):
        """
        存储为tfrecords格式的数据
        :param label_batch: 样本标签
        :param image_batch: 样本特征
        :return: None
        """
        # 创建tfrecord存储器
        tf_writer = tf.python_io.TFRecordWriter('../tmp/tfrecords/cifar.tfrecords')
        # 将每个样本分别写入tfrecords文件
        for i in range(20):
            # 取出第i条数据的标签和特征。eval()只能在会话中使用
            # label是二维数组，需要取出单独的元素值
            label = label_batch[i].eval()[0]
            image = image_batch[i].eval().tostring()
            # 构造每个样本的Example协议块。类字典格式的数据，'image'是键，value=[image]是值
            example = tf.train.Example(features=tf.train.Features(feature={
                # BytesList和Int64List都是序列化类型，还有FloatList
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            # 序列化并写出数据
            tf_writer.write(example.SerializeToString())

        # 关闭资源
        tf_writer.close()
        return None

    def read_from_tfrecords(self, rec_file_list):
        """
        加载tfrecords文件
        :param rec_file_list: 文件列表
        :return: image_batch 批处理的特征, label_batch 批处理的标签
        """
        # 创建文件列表队列
        file_queue = tf.train.string_input_producer(rec_file_list)
        # 创建文件阅读器，加载文件
        reader = tf.TFRecordReader()
        # key是文件名，value是数据
        key, value = reader.read(file_queue)
        # 解析Example（反序列化）
        example = tf.parse_single_example(value, features={
            # 'image'是Example的键，shape为输入数据的形状，一般不指定。dtype是反序列化类型
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.FixedLenFeature(shape=[], dtype=tf.int64)
        })
        # 解码。如果读取的内容是string类型的需要解码，int和float类型不需要解码
        image = tf.decode_raw(example['image'], tf.uint8)
        label = tf.cast(example['label'], tf.int32)
        print(image, label)
        # 固定形状（统一特征维度）
        image_reshape = tf.reshape(image, [self.height, self.width, self.channels])
        # 批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=20, num_threads=1, capacity=20)
        return image_batch, label_batch


if __name__ == '__main__':
    # 1.找到文件，创建文件列表
    file_names = os.listdir('../data/cifar-10-binary')
    file_list = [os.path.join('../data/cifar-10-binary', file_name) for file_name in file_names if
                 file_name[-3:] == 'bin']
    tfrecords = tfrecords_file(file_list)
    label_batch, image_batch = tfrecords.read_and_decode()

    # 加载tfrecords数据
    rec_file_names = os.listdir('../tmp/tfrecords/')
    rec_file_list = [os.path.join('../tmp/tfrecords/', file_name) for file_name in rec_file_names]
    image, label = tfrecords.read_from_tfrecords(rec_file_list)

    # 会话运行程序
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()
        # 启动加载数据的线程
        threads = tf.train.start_queue_runners(sess, coord)
        # print(sess.run([label_batch, image_batch]))

        # 将数据存储为tfrecords格式
        # print('开始存储...')
        # tfrecords.save_to_tfrecords(label_batch, image_batch)
        # print('存储完毕...')

        # 查看tfrecords文件内容
        print(sess.run([image, label]))

        # 等待线程结束
        coord.request_stop()
        coord.join(threads)
