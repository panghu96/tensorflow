import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(3)
b = tf.constant(5)
sum1 = tf.add(a, b)

# 获取默认的图
graph = tf.get_default_graph()
print(graph)
# config=... 开启交互式
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(a.shape)
    print(sum1.eval())

# 创建新的图
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(3)
    print(c)
    print(a)

print('-------------------------------------------------------')
# placeholder：占位符，后面可以通过feed_dict覆盖当前张量的值
t1 = tf.placeholder(dtype=tf.float32)
t2 = tf.placeholder(dtype=tf.float32)
res = tf.add(t1, t2)
t3 = tf.placeholder(dtype=tf.int32, shape=[2, 3])

with tf.Session() as sess:
    print(sess.run(res, feed_dict={t1: 1.0, t2: 3.0, t3: [[1, 2, 3], [4, 5, 6]]}))
    # 张量的图
    print(res.graph)
    # 张量的形状
    print(t3.shape)
    # 张量的描述
    print(res.name)
    # 张量的操作名
    print(res.op)

print('--------------------------------------------------------')

# 不定形状的张量
t4 = tf.placeholder(dtype=tf.int32, shape=[None, 2])
with tf.Session() as sess:
    print(sess.run(t4, feed_dict={t4: [[1, 2], [3, 4], [5, 6]]}))
    print(t4.shape)
    print(t4.op)

print('--------------------------------------------------------')

# 修改张量的形状
t5 = tf.placeholder(dtype=tf.int32, shape=[None, 4])

with tf.Session() as sess:
    sess.as_default()
    # 静态形状，不生成新的张量。静态形状不能跨维度改变
    tf.Tensor.set_shape(t5, [3, 4])
    print(sess.run(t5, {t5: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]}))
    print(t5.shape)
    print(tf.Tensor.get_shape(t5))
    # 静态形状一旦指定后，不可以再重新设定
    # tf.Tensor.set_shape(t5, [4,3])
    # 动态形状，生成新的张量
    t6 = tf.reshape(t5, [4, 3])
    print(t6.shape)
    # 动态形状可以跨维度改变，但是元素个数要相等
    t7 = tf.reshape(t6, [3, 2, 2])
    print(t7.shape)
    print(t7)

print('--------------------------------------------------------')
