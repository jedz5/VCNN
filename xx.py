import tensorflow as tf
from tensorflow.python.training import moving_averages
w1 = tf.Variable(3.0)
w2 = tf.Variable(4.0)
ema_val1 = tf.Variable(100.0)
ema_val2 = tf.Variable(200.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update1 = tf.assign_add(w1,1.0)
update2 = tf.assign_add(w2,1.0)

with tf.control_dependencies([update1,update2]):
    ema_op = ema.apply([w1,w2])#这
    #ema_op1 = moving_averages.assign_moving_average(ema_val1, w1, 0.5)
    #ema_op2 = moving_averages.assign_moving_average(ema_val2, w2, 0.5)
with tf.control_dependencies([ema_op]):
    ema_val1 = ema.average(w1)#参数不能是list，有点蛋疼
    ema_val2 = ema.average(w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        # sess.run(ema_op)
        # print(sess.run(ema_val1))
        #
        # print(sess.run(ema_val2))
        print('ema_op1=',sess.run(ema_op))
        #print('ema_op2=',sess.run(ema_op2))
        print('ema_val1=', sess.run(ema_val1))
        print('ema_val2=', sess.run(ema_val2))
        print('w1=',sess.run(w1))
        print('w2=',sess.run(w2))
# 创建一个时间序列 1 2 3 4
#输出：
#1.1      =0.9*1 + 0.1*2
#1.29     =0.9*1.1+0.1*3
#1.561    =0.9*1.29+0.1*4

# pred = tf.placeholder(tf.bool)
# x = tf.Variable(1)
# # def update_x_2():
# #   with tf.control_dependencies([tf.assign(x, 2)]):
# #     return tf.identity(x)
# def x2():
#     return tf.assign(x, 2)
# with tf.control_dependencies([x2()]):
#     y = tf.cond(pred, x2, lambda: x)
# with tf.Session() as session:
#     session.run(tf.initialize_all_variables())
#   # print(y.eval(feed_dict={pred: False}))  # ==> [1]
#   # print(y.eval(feed_dict={pred: True}))   # ==> [2]
#
#     print(session.run(y,feed_dict={pred: False}))  # ==> [1]
#     print(session.run(y,feed_dict={pred: True}))   # ==> [2]