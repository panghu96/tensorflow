"""
自定义实现线性回归
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def regression():
    # 创建指定名字的作用域
    with tf.variable_scope("data"):
        # 1.随机生成数据，x和y都必须是矩阵
        x = tf.random_normal([100, 1], mean=1.70, stddev=0.50, name='x_data')
        # 矩阵相乘，计算y. y = wx + b
        y = tf.matmul(x, [[0.7]]) + 0.9

    with tf.variable_scope("train"):
        # 2.定义随机初始w和b。w和b必须是变量,trainable默认True，表示数据跟随梯度下降变化
        w = tf.Variable(tf.random_normal([1, 1]), name='w', trainable=True)
        b = tf.Variable(tf.random_normal([1]),  name='b')
        # 变量显式初始化op
        init_op = tf.global_variables_initializer()

    with tf.variable_scope("loss"):
        # 3.计算损失
        y_predict = tf.matmul(x, w) + b
        sqr = tf.square(y - y_predict)
        loss = tf.reduce_mean(sqr, name='loss')

    with tf.variable_scope("optimizer"):
        # 4.梯度下降，使得损失最小。learning_rate学习率，一般取0-1之间
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    # 收集tensor
    tf.summary.scalar("losses", loss)   # 收集标量
    tf.summary.histogram("weight", w)      # 收集高维变量
    # 合并tensor
    merged = tf.summary.merge_all()

    # 保存模型,参数为要保存和还原的变量
    saver = tf.train.Saver([w, b])

    # 通过会话运行程序
    with tf.Session() as sess:
        # 运行初始化op
        sess.run(init_op)
        # 还原变量
        saver.restore(sess, '/home/sun/py_code/py36/deep_learning/day01/tmp/ckpt/model')
        # 保存事件文件
        filewriter = tf.summary.FileWriter('/home/sun/py_code/py36/deep_learning/day01/tmp/summary/test', graph=sess.graph)
        print('初始权重为：%f，偏置为：%f，损失为：%f' % (w.eval(), b.eval(), loss.eval()))
        # 循环迭代，使得损失最小
        for i in range(500):
            sess.run(train_op)
            # 运行合并op，写入事件文件
            summary = sess.run(merged)
            # i表示每一次的值
            filewriter.add_summary(summary, i)
            print('第%d次梯度下降后权重为：%f，偏置为：%f，损失为：%f' % (i, w.eval(), b.eval(), loss.eval()))
        # 保存模型训练结果
        saver.save(sess, '/home/sun/py_code/py36/deep_learning/day01/tmp/ckpt/model')
    return None


if __name__ == '__main__':
    regression()
