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
learning_rate = 0.05
training_iters = 20000
#batch_size = 128
display_step = 10

# Network Parameters
hexY = lj.side
hexX = lj.stacks
hexDepth = lj.hexDepth
n_classes = lj.n_classes
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, hexY,hexX,hexDepth])
y = tf.placeholder(tf.float32, [None, n_classes,1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
is_train = tf.placeholder(tf.bool)
fc1 = 20
fc2 = 20
# Store layers weight & bias
weights = {
    # fully connected, 7*7*64 inputs, 1024 outputs
    'fc1': tf.Variable(tf.random_normal([hexY*hexX*hexDepth, fc1])),
    #'fc2': tf.Variable(tf.random_normal([fc1, fc2])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([fc1, n_classes]))
}

biases = {
    'fc1': tf.Variable(tf.random_normal([fc1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
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
    x = tf.reshape(x, shape=[-1,2*7*7])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(x, weights['fc1'])+biases['fc1']
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.matmul(fc1, weights['out'])+biases['out']
    return out

def train(batch_x,batch_y):
    # Construct model
    pred = conv_net(x,keep_prob)

    # Define loss and optimizer
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(tf.round(pred)*x_ - y)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Launch the graph

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step  < 800:
            # Run optimization op (backprop)
            sess.run([optimizer], feed_dict={x: batch_x[:800], y: batch_y[:800],
                                           keep_prob: dropout,is_train:True})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x[:800],
                                                                  y: batch_y[:800],
                                                                  keep_prob: 1.,is_train:False})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", 训练准确率= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("训练结束")
        saver.save(sess, './result/model.ckpt')

    # with tf.Session() as sess:
    #     sess.run(init)
    #     saver.restore(sess, tf.train.latest_checkpoint('./NN1'))
    #     res = sess.run(accuracy, feed_dict={x: batch_x[801:], y: batch_y[801:],keep_prob: 1.,is_train:False})
    #     print("测试准确率:",res)

