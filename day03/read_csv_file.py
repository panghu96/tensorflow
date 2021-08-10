"""
tensorflow读取csv文件
1.找到文件，指定文件列表。路径+文件名
2.创建文件列表队列。
3.创建文件阅读器，读取文件，返回的是k-v类型，分别是：文件名-文件内容。默认只读取一行。
4.解码，转为tensor张量，指定每一列的数据类型和默认值。
5.批处理。
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_csv(file_list):
    # 2.创建文件队列
    file_queue = tf.train.string_input_producer(file_list)
    # 3.创建阅读器，读取数据，默认只读取一行
    reader = tf.TextLineReader()
    # 返回的是k-v类型，k是文件名字，v是文件内容
    key, value = reader.read(file_queue)
    # 4.解码，转为tensor张量
    # record_defaults 指定每一列的类型和默认值
    records = [['None'], ['None']]
    field1, field2 = tf.decode_csv(value, record_defaults=records)
    # 5.批处理。batch_size 批处理的样本数。num_threads 进行操作的线程数。capacity 队列容量
    field1_batch, field2_batch = tf.train.batch([field1, field2], batch_size=9, num_threads=1, capacity=9)
    return field1_batch, field2_batch


if __name__ == '__main__':
    # 1.找到文件，构建文件列表
    file_names = os.listdir('../data/text/')
    # 路径+文件名
    file_list = [os.path.join('../data/text/', file_name) for file_name in file_names]
    field1_batch, field2_batch = read_csv(file_list)
    # 会话运行程序
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()
        # 启动读取数据线程
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run([field1_batch, field2_batch]))
        # 等待线程结束，关闭线程
        coord.request_stop()
        coord.join(threads)