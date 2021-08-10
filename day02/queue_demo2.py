"""
模拟实现异步策略
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def asynchronous():
    # 1.定义队列
    queue = tf.FIFOQueue(1000, tf.float32)
    # 2.定义变量
    var = tf.Variable(0.0)
    # 3.定义子线程操作。入队
    # 实现变量自增，每次自增1.0
    data = tf.assign_add(var, tf.constant(1.0))
    que_en = queue.enqueue(data)
    # 定义队列管理器。参数1：队列。参数2：线程操作列表，*2表示两个线程
    qr = tf.train.QueueRunner(queue, [que_en] * 2)
    # 变量初始化op
    init_op = tf.global_variables_initializer()
    # 会话运行
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init_op)
        # 创建线程协调器
        coord = tf.train.Coordinator()
        # 启用子线程
        threads = qr.create_threads(sess, coord=coord, start=True)
        # 主线程读取数据
        for i in range(300):
            print(sess.run(queue.dequeue()))
        # 等待线程结束
        coord.request_stop()
        coord.join(threads)
    return None


if __name__ == '__main__':
    asynchronous()
