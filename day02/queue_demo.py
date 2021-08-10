"""
使用队列，模拟实现同步策略
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def synchronized():
    # 创建队列
    queue = tf.FIFOQueue(3, tf.float32)
    # 入队
    que_many = queue.enqueue_many([[0.1, 0.2, 0.3], ])
    # 出队，进行+1操作
    de_queue = queue.dequeue()
    # 运算可以重载，op自动转换
    en_val = de_queue + 1
    en_queue = queue.enqueue(en_val)
    # 会话运行
    with tf.Session() as sess:
        # 初始化队列
        sess.run(que_many)
        # 处理队列的数据
        for i in range(300):
            # tensorflow的依赖机制，只需运行依赖最后的op即可
            sess.run(en_queue)
        # 出队，只有当队列中的数据处理完毕，才会进行下面的操作。
        for i in range(queue.size().eval()):
            print(sess.run(queue.dequeue()))

    return None


if __name__ == '__main__':
    synchronized()
