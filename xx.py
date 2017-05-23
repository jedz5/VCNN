import tensorflow as tf
# w1 = tf.Variable(3.0)
# w2 = tf.Variable(4.0)
# ema_val1 = tf.Variable(100.0,trainable=False)
# ema_val2 = tf.Variable(200.0,trainable=False)
# ema = tf.train.ExponentialMovingAverage(0.9)
# add1 = tf.assign_add(w1,1.0)
# add2 = tf.assign_add(w2,1.0)
# ema_op = ema.apply([w1,w2])#这
# with tf.control_dependencies([add1,add2,ema_op]):
#     ema_val1 = ema.average(w1)
#     ema_val2 = ema.average(w2)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for i in range(3):
#         # sess.run(ema_op)
#         print("before",sess.run(w1))
#         print(sess.run(w2))
#         print(sess.run(ema_val1))
#         print(sess.run(ema_val2))
#         print(sess.run(w1))
#         print("after",sess.run(w2))

w1 = tf.Variable(tf.ones([5]))
w2 = tf.Variable(tf.ones([5]))
one = tf.Variable(tf.ones([5]))
decay = 0.9
ema_val1 = tf.Variable(100.0,trainable=False)
ema = tf.train.ExponentialMovingAverage(0.9)
def X(weight):
    add = tf.assign_add(w1,one)
    ema_op = ema.apply([w1])
    tf.add_to_collection("updateEMA",[add,ema_op])
# op = tf.group(add,ema_op)
ema_val1 = ema.average(w1)
#     ema_val2 = ema.average(w2)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    op = tf.get_collection("updateEMA")
    for i in range(3):
        print("before",sess.run(w1))
        print(sess.run(op))
        print(sess.run(ema_val1))
        print("after",sess.run(w1))


    # ws = tf.trainable_variables()
    # for w in ws:
    #     print(w.name)
# def test(name=None):
#     w = tf.Variable([2, 10])
# def st():
#         test()
#         test()
# for x in range(3):
#     st()
# ws = tf.trainable_variables()
# for w in ws:
#     print(w.name)

# w = tf.Variable(1)
# mul = tf.multiply(w, 2)
# add = tf.add(w, 2)
# group = tf.group(mul,add)
# tuple = tf.tuple([mul, add])
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(group))
#     print(sess.run(tuple))

# def foo(*args, **kwargs):
#     print('args = ', args)
#     print('kwargs = ', kwargs)
#     print('---------------------------------------')
#
# if __name__ == '__main__':
#     # foo(1,2,3,4)
#     # foo(a=1,b=2,c=3)
#     kwargs = {'a': 4, 'c': 3, 'b': '2'}
#     foo(1,2,3,4, **kwargs)
#     # foo('a', 1, None, a=1, b='2', c=3)
