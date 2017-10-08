'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
# from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers.python.layers import utils
# Parameters
learning_rate = 0.05
training_iters = 20000
#batch_size = 128
display_step = 10

# Network Parameters
hexY = 11
hexX = 17
hexDepth = 16
n_classes = 8
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, hexY,hexX,hexDepth])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
is_train = tf.placeholder(tf.bool)
ema = tf.train.ExponentialMovingAverage(0.999)
fk1 = 3
fk2 = 3
c1 = 16
c2 = 16
fc1 = 64
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'bnM0': tf.Variable([hexDepth],trainable=False),
    'bnV0': tf.Variable([hexDepth],trainable=False),
    'wc1': tf.Variable(tf.random_normal([fk1, fk1, hexDepth, c1])),
    'bnM1': tf.Variable([c1],trainable=False),
    'bnV1': tf.Variable([c1],trainable=False),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([fk2, fk2, c1,c2])),
    'bnM2': tf.Variable([c2],trainable=False),
    'bnV2': tf.Variable([c2],trainable=False),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([hexY*hexX*c2, fc1])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([fc1, n_classes]))
}

# biases = {
#     'bc1': tf.Variable(tf.random_normal([c1])),
#     'bc2': tf.Variable(tf.random_normal([c2])),
#     'bd1': tf.Variable(tf.random_normal([fc1])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
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
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # with tf.variable_scope("batch_norm1") as bn1:
    #     bx = batch_norm(x)
    # Convolution Layer
    x = conv2d(x, weights['wc1'])
    # with tf.variable_scope("batch_norm2") as bn2:
    #     bc1 = batch_norm(bx)

    # Convolution Layer
    x = conv2d(x, weights['wc2'])
    # with tf.variable_scope("batch_norm3") as bn3:
    #     bc2 = batch_norm(bc1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1'])
    fc1 = tf.layers.batch_normalization(fc1,training=is_train)
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.matmul(fc1, weights['out'])
    out = tf.layers.batch_normalization(out, training=is_train)
    return out



# def batch_norm(inputs,is_conv_out=True,decay = 0.5):
#
#     scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
#     beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
#     batch_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
#     batch_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
#     def in_train(batch_mean,batch_var):
#         if is_conv_out:
#             temp_mean,temp_var = tf.nn.moments(inputs,[0,1,2])
#             assM = tf.assign(batch_mean,temp_mean)
#             assV = tf.assign(batch_var,temp_var)
#         else:
#             temp_mean, temp_var = tf.nn.moments(inputs, [0])
#             assM = tf.assign(batch_mean, temp_mean)
#             assV = tf.assign(batch_var, temp_var)
#         with tf.control_dependencies([assM,assV]):
#             train_mean = ema.apply([batch_mean,batch_var])
#         # tf.add_to_collection("updateEMA", train_mean)
#         print(ema.average_name(batch_mean))
#         print(ema.average_name(batch_var))
#         with tf.control_dependencies([train_mean]):
#             return tf.nn.batch_normalization(inputs,
#                     batch_mean, batch_var, beta, scale, 0.001)
#     def in_val():
#         # op = tf.get_collection("updateEMA")
#         # with tf.control_dependencies(op):
#         return tf.nn.batch_normalization(inputs,ema.average(batch_mean), ema.average(batch_var), beta, scale, 0.001)
#     return utils.smart_cond(is_train,in_train(batch_mean,batch_var),in_val())
def train(batch_x,batch_y):
    # Construct model
    pred = conv_net(x,keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step  < 5000:
            # Run optimization op (backprop)
            sess.run([optimizer], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout,is_train:True})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.,is_train:False})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                                   y: mnist.test.labels[:256],
        #                                   keep_prob: 1.}))

