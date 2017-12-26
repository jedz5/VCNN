'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import loadJson as lj
from tensorflow.python.ops import control_flow_ops
# from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers.python.layers import utils
# Parameters
learning_rate = 0.005
training_iters = 20000
#batch_size = 128
display_step = 5

# Network Parameters
hexY = lj.side
stacks = lj.stacks
hexDepth = lj.hexDepth
n_all = lj.n_all
dropout = 0.5 # Dropout, probability to keep units
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_all])
x_M = tf.placeholder(tf.float32, [None,1])
y_C = tf.placeholder(tf.float32, [None, stacks])
y_M = tf.placeholder(tf.float32, [None,1])
in_amout = tf.placeholder(tf.float32, [None, stacks])
in_value = tf.placeholder(tf.float32, [None, stacks])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
is_train = tf.placeholder(tf.bool)
fcn1 = 40
fcn2 = 40
# Store layers weight & bias
weights = {
    # fully connected, 7*7*64 inputs, 1024 outputs
    'fc1': tf.Variable(tf.random_normal([n_all, fcn1])),
    'fc2': tf.Variable(tf.random_normal([fcn1, fcn2])),
    # 1024 inputs, 10 outputs (class prediction)
    'casul': tf.Variable(tf.random_normal([fcn2, stacks])),
    'mana': tf.Variable(tf.random_normal([fcn2, 1]))
}

biases = {
    'fc1': tf.Variable(tf.random_normal([fcn1])),
    'fc2': tf.Variable(tf.random_normal([fcn2])),
    'casul': tf.Variable(tf.random_normal([stacks])),
    'mana': tf.Variable(tf.random_normal([1]))
}
# Create some wrappers for simplicity
def  conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.layers.batch_normalization(x,training=is_train)
    return tf.nn.relu(x)


# def maxpool2d(x, k=2):
#     # MaxPool2D wrapper
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')

# Create model
def conv_net(x,dropout):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1,2*7*7])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(x, weights['fc1'])+biases['fc1']
    fc1 = tf.layers.batch_normalization(fc1, training=is_train)
    fc1 = tf.nn.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    fc1 = tf.matmul(fc1, weights['fc2']) + biases['fc2']
    fc1 = tf.layers.batch_normalization(fc1, training=is_train)
    fc1 = tf.nn.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    casul = tf.matmul(fc1, weights['casul'])+biases['casul'] #每个slot的伤亡比例
    casul = tf.layers.batch_normalization(casul, training=is_train)
    casul = tf.nn.sigmoid(casul)
    mana = tf.matmul(fc1, weights['mana']) + biases['mana'] #魔法消耗比例
    mana = tf.layers.batch_normalization(mana, training=is_train)
    mana = tf.nn.sigmoid(mana)
    return casul,mana

if __name__ == '__main__':
    bx, bxm, byc, bym, b_amount, b_value = lj.loadData(".")
    mPercent = bym / bxm
    # Construct model
    predC,predM = conv_net(x,keep_prob)
    calsu = tf.round(predC * in_amout)
    lossC1 = tf.reduce_sum(calsu *in_value,1) #每个slot比例*总数取整 再*aiValue
    lossC2 = tf.reduce_sum(y_C*in_value,1)
    lossC = tf.abs(lossC1 - lossC2)/lossC2
    lossM = tf.abs(predM - y_M)
    accuracyC = tf.reduce_mean(lossC)
    accuracyM = tf.reduce_mean(lossM)
    cost = (accuracyC+accuracyM)/2
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step  < 50:
            # Run optimization op (backprop)
            sess.run([optimizer], feed_dict={x: bx,
                                             y_C: byc,
                                             y_M:mPercent,
                                             in_amout:b_amount,
                                             in_value:b_value,
                                           keep_prob: dropout,is_train:True})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([accuracyC, accuracyM], feed_dict={x: bx,
                                             y_C: byc,
                                             y_M:mPercent,
                                             in_amout:b_amount,
                                             in_value:b_value,
                                           keep_prob: 1,is_train:False})
                print("Iter " + str(step) + ", 伤亡误差= " + \
                      "{:.6f}".format(loss) + ", mana误差= " + \
                      "{:.6f}".format(acc))
            step += 1
        print("训练结束")
        for n in range(5, 12):
            cas,mc,loss, acc = sess.run([calsu,predM,accuracyC, accuracyM], feed_dict={x: bx[n:n+1],
                                                                y_C: byc[n:n+1],
                                                                y_M: mPercent[n:n+1],
                                                                in_amout: b_amount[n:n+1],
                                                                in_value: b_value[n:n+1],
                                                                keep_prob: 1, is_train: False})
            print("iter: ",n)
            print(byc[n:n+1],mPercent[n:n + 1])
            print(cas,mc)
            print(np.floor(b_value[n:n + 1]))
            print("伤亡误差= " + \
                  "{:.6f}".format(loss) + ", mana误差= " + \
                  "{:.6f}".format(acc))
            saver.save(sess, './result/model.ckpt')

    # with tf.Session() as sess:
    #     sess.run(init)
    #     saver.restore(sess, tf.train.latest_checkpoint('./NN1'))
    #     loss, acc = sess.run([accuracyC, accuracyM], feed_dict={x: bx,
    #                                                             y_C: byc,
    #                                                             y_M: mPercent,
    #                                                             in_amout: b_amount,
    #                                                             in_value: b_value,
    #                                                             keep_prob: dropout, is_train: True})
    #     print("测试准确率 " + str(step) + ", 伤亡准确率= " + \
    #           "{:.6f}".format(loss) + ", mana准确率= " + \
    #           "{:.6f}".format(acc))
       # print("测试准确率:",res)

