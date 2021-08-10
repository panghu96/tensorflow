"""
自定义命令行参数
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 自定义命令行参数
# flag_name 参数名字    default_value 默认值   docstring 描述
tf.app.flags.DEFINE_integer(flag_name='step', default_value=200, docstring='训练次数')
FLAGS = tf.app.flags.FLAGS


def regression():
    # 1.生成数据
    with tf.name_scope("data"):
        x = tf.random_normal([100, 1], mean=1.7, stddev=0.25, name='x_data')
        y = tf.matmul(x, [[0.7]]) + 0.9

    # 2.随机初始权重和偏置
    with tf.name_scope("train"):
        w = tf.Variable(tf.random_normal([1, 1]))
        b = tf.Variable(tf.random_normal([1]))
        y_predict = tf.matmul(x, w) + b

    # 3.计算损失
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y - y_predict), name='loss')

    # 4.梯度下降
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

    # 变量初始化op
    init_op = tf.global_variables_initializer()
    # 通过会话运行
    with tf.Session() as sess:
        # 运行变量初始化op
        sess.run(init_op)
        print('初始权重：%f，偏置：%f, 损失：%f' % (w.eval(), b.eval(), loss.eval()))
        # 梯度下降优化损失
        # 获取命令行参数
        for i in range(FLAGS.step):
            sess.run(optimizer)
            print('第%d次优化后，权重：%f，偏置：%f，损失：%f' % (i, w.eval(), b.eval(), loss.eval()))
    return None


if __name__ == '__main__':
    regression()
