import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# name用于tensorboard中显示名字，区分相同的op
a = tf.constant(3.0, name='a')
b = tf.constant(4.0, name='b')
c = tf.add(a, b, name='add')
# 定义变量,值是随机生成的
var = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1.0), name='var')
# 变量必须要进行显示初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)
    # 保存事件文件，写的是文件目录
    fileWriter = tf.summary.FileWriter('./tmp/tensorflow/test', graph=sess.graph)
    print(sess.run([c, var]))
