"""
tensorflow读取图片文件
1.找到文件，指定文件列表
2.创建文件列表队列
3.创建阅读器，读取文件。
4.解码，统一图片大小（统一特征），固定形状
5.批处理
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_images(file_list):
    # 2.创建文件列表队列
    file_queue = tf.train.string_input_producer(file_list)
    # 3.创建文件阅读器，加载文件
    reader = tf.WholeFileReader()
    # 返回k-v类型的数据，key是文件名称，value是文件内容
    key, value = reader.read(file_queue)
    # 4.解码，统一图片大小（特征数量）
    image = tf.image.decode_jpeg(value)
    image_resize = tf.image.resize_images(image, size=[200, 200])
    # 批处理必须指定形状，[长, 宽（像素）, 3代表RGB通道|1代表灰度通道]
    image_resize.set_shape([200, 200, 3])
    # 5.批处理，数据必须是列表形式
    image_batch = tf.train.batch([image_resize], batch_size=10, num_threads=1, capacity=10)
    return image_batch


if __name__ == '__main__':
    # 1.找到文件，创建文件列表
    file_names = os.listdir('../data/images/')
    file_list = [os.path.join('../data/images/', file_name) for file_name in file_names]
    image_batch = read_images(file_list)
    # 会话运行程序
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()
        # 启动读取数据的线程
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run(image_batch))
        # 等待线程结束，关闭线程
        coord.request_stop()
        coord.join(threads)
