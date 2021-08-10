"""
只有全连接层的神经网络
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 自定义命令行参数
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('is_train', 1, "是否训练模型")


def full_connect():
    # 加载数据
    data = input_data.read_data_sets('../data/mnist/', one_hot=True)
    # 1.创建占位符，使用时再填充数据。x [None, 784]  y [None, 10]
    with tf.variable_scope('data'):
        # None表示样本数量不固定
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='y_true')

    # 2.初始化权重和偏置。matrix运算，[None, 784] * [784, 10] + [10] = [None, 10]
    with tf.variable_scope('predict'):
        weight = tf.Variable(tf.random_normal(shape=[784, 10], mean=0.0, stddev=1.0), name='w')
        bias = tf.Variable(tf.constant(0.0, shape=[10]), name='b')
        # 预测结果
        y_predict = tf.matmul(x, weight) + bias
    # 3.softmax计算结果，并计算交叉熵损失平均值
    with tf.variable_scope('softmax_cross'):
        cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        loss = tf.reduce_mean(cross)
    # 4.反向传播优化(梯度下降)
    with tf.variable_scope('optimizer'):
        # learning_rate学习率
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 5.计算准确率。原理：比较每个样本的真实分类索引和预测分类概率最大的索引是否相等，再求相等的占总数的比例
    with tf.variable_scope('accu'):
        # axis=1表示取每行最大的值，即每个样本的真实分类和预测的最大概率
        equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量
    # 收集标量
    tf.summary.scalar('losses', loss)
    tf.summary.scalar('accu', accuracy)
    # 收集高维变量
    tf.summary.histogram("weights", weight)
    tf.summary.histogram("biases", bias)
    # 合并收集的变量
    merge = tf.summary.merge_all()

    # 创建saver，保存模型
    saver = tf.train.Saver()
    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # 会话运行程序
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init_op)
        # 创建events文件，写入数据
        filewriter = tf.summary.FileWriter('../tmp/summary/shallow/', graph=sess.graph)

        # 判断是否要训练模型
        if FLAGS.is_train == 1:
            # 迭代训练，占位符赋值
            for i in range(2000):
                # 取出x和y，每批次取50个样本
                x_mnist, y_mnist = data.train.next_batch(50)
                # 注意run所有和数据相关的op都需要填充值
                sess.run(train_op, feed_dict={x: x_mnist, y_true: y_mnist})

                # 将每次的结果写入events文件
                summary = sess.run(merge, feed_dict={x: x_mnist, y_true: y_mnist})
                filewriter.add_summary(summary, i)
                print('第%d次训练，准确率为%f' % (i, sess.run(accuracy, feed_dict={x: x_mnist, y_true: y_mnist})))

            # 保存训练模型。保存路径必须存在
            saver.save(sess, '../tmp/ckpt/')
        else:
            # 加载模型
            saver.restore(sess, '../tmp/ckpt/')
            # 判断测试数据集中100张图片
            for i in range(100):
                # 加载测试集特征和标签
                x_test, y_test = data.test.next_batch(1)
                # 预测
                print('第%d张图片，手写目标值为%d，预测目标值为%d' % (
                    i,
                    # 取出one_hot编码不为0的那一项的索引值，即为类别值
                    tf.argmax(y_test, axis=1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), axis=1).eval()
                ))
    return None


if __name__ == '__main__':
    full_connect()
