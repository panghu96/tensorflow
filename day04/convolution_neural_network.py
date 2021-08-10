"""
卷积神经网络，对手写数字进行识别
卷积层1
    卷积：filter32个 长5，宽5，步长1，padding="SAME"
    输入  [None, 784] => [None,28,28,1] 输出 [None,28,28,32]
    激活函数relu
    池化 窗口长2，宽2，步长2 输入 [None,28,28,32] 输出 [None,14,14,32]
卷积层2
    卷积：filter64个，长5，宽5，步长1
    输入 [None,14,14,32] 输出 [None,14,14,64]
    激活函数
    池化 长2，宽2，步长2，输入[None,14,14,64] 输出[None,7,7,64]
全连接层
    输入[None,7,7,64]=> [None,7*7*64] 输出 [None, 10]
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_weight(shape):
    w = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义卷积模型
    :return:
    """
    # 创建数据占位符
    with tf.variable_scope('conv1'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None,10])

    # 卷积层1
    with tf.variable_scope('conv1'):
        # 改变x的形状  [None, 784] => [None,28,28,1]
        x_reshape_conv1 = tf.reshape(x, [-1, 28, 28, 1])
        # 定义随机权重(过滤器)和偏置  权重：指定过滤器 filter32个 长5，宽5 [长，宽，输入通道，输出通道]
        w_conv1 = get_weight([5, 5, 1, 32])
        b_conv1 = get_bias([32])
        # 卷积 strides=[1,步长,步长,1] padding='SAME'启用0填充
        tf_conv1 = tf.nn.conv2d(x_reshape_conv1, filter=w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        # 激活函数
        relu_conv1 = tf.nn.relu(tf_conv1)
        # 池化  长2，宽2，步长2 ksize:池化窗口大小，[1, ksize, ksize, 1] strides:步长大小，[1,strides,strides,1]
        pool_conv1 = tf.nn.max_pool(relu_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层2
    with tf.variable_scope('conv2'):
        # 定义随机权重(过滤器)和偏置  权重：指定过滤器 filter64个，长5，宽5，步长1 [长，宽，输入通道，输出通道]
        w_conv2 = get_weight([5, 5, 32, 64])
        b_conv2 = get_bias([64])
        # 在卷积1的结果上继续卷积
        tf_conv2 = tf.nn.conv2d(pool_conv1, filter=w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        # 激活函数
        relu_conv2 = tf.nn.relu(tf_conv2)
        # 池化 长2，宽2，步长2  输出[None,7,7,64]
        pool_conv2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    with tf.variable_scope('fc_connect'):
        # 输入[None, 7, 7, 64] => [None,7*7*64] 输出 [None, 10]
        x_reshape_fc = tf.reshape(pool_conv2, [-1, 7*7*64])
        # 定义随机权重和偏置
        w_fc = get_weight([7*7*64, 10])
        b_fc = get_bias([10])
        # 预测
        y_predict = tf.matmul(x_reshape_fc, w_fc) + b_fc

    return x, y_true, y_predict

def conv_fc():
    # 加载数据
    data = input_data.read_data_sets('../data/mnist/', one_hot=True)
    # 加载模型返回的数据
    x, y_true, y_predict = model()
    # 3.softmax计算结果，并计算交叉熵损失平均值
    with tf.variable_scope('softmax_cross'):
        cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        loss = tf.reduce_mean(cross)
    # 4.反向传播优化(梯度下降)
    with tf.variable_scope('optimizer'):
        # learning_rate学习率
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

    # 5.计算准确率。原理：比较每个样本的真实分类索引和预测分类概率最大的索引是否相等，再求相等的占总数的比例
    with tf.variable_scope('accu'):
        # axis=1表示取每行最大的值，即每个样本的真实分类和预测的最大概率
        equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量op
    init_op = tf.global_variables_initializer()
    # 会话运行程序
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init_op)
        # 迭代训练，占位符赋值
        for i in range(2000):
            # 取出x和y，每批次取50个样本
            x_mnist, y_mnist = data.train.next_batch(50)
            # 注意run所有和数据相关的op都需要填充值
            sess.run(train_op, feed_dict={x: x_mnist, y_true: y_mnist})
            print('第%d次训练，准确率为%f' % (i, sess.run(accuracy, feed_dict={x: x_mnist, y_true: y_mnist})))
    return None


if __name__ == '__main__':
    conv_fc()
