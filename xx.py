import tensorflow as tf

# pred = tf.constant(True)
# x = tf.Variable([1])
# def assign_x_2():
#     print("3333")
#     tf.assign(x, [2])
#     return x
# def fs():
#     print("4444")
#     tf.assign(x, [3])
#     return x
# # def update_x_2():
# #   with tf.control_dependencies([assign_x_2]):
# #     return tf.identity(x)
# y = tf.cond(pred, assign_x_2, fs)
# with tf.Session() as session:
#   session.run(tf.global_variables_initializer())
#   print("1111")
#   session.run(y,feed_dict={pred: False})  # ==> [1]
#   print(y.eval())
#   #print(y.eval(feed_dict={pred: True}))  # ==> [2]
#   session.run(y, feed_dict={pred: True})
#   print(y.eval())



# pred = tf.Variable([True])
# x = tf.Variable([1])
# def update_x_2():
#   with tf.control_dependencies([tf.assign(x, [2])]):
#     return tf.identity(x)
# y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
# with tf.Session() as session:
#   session.run(tf.initialize_all_variables())
#   session.run(y,feed_dict={pred: False})
#   print(y.eval())  # ==> [1]
#   session.run(y, feed_dict={pred: True})
#   print(y.eval())   # ==> [2]

pred = tf.placeholder(tf.bool, shape=[])
x = tf.Variable([1])
def update_x_2():
  with tf.control_dependencies([tf.assign(x, [2])]):
    return tf.identity(x)
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
  # print(y.eval(feed_dict={pred: False}))  # ==> [1]
  # print(y.eval(feed_dict={pred: True}))   # ==> [2]
    session.run(y,feed_dict={pred: False})
    print(y.eval())  # ==> [1]
    session.run(y, feed_dict={pred: True})
    print(y.eval())   # ==> [2]