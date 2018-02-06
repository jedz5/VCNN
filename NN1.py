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
#learning_rate = 0.02
training_iters = 5000
#batch_size = 128
display_step = 50

# Network Parameters
hexY = lj.side
stacks = lj.stacks
hexDepth = lj.hexDepth
n_all = lj.n_all
dropout = 0.5 # Dropout, probability to keep units
epsino = 60
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_all])
in_M = tf.placeholder(tf.float32, [None,1])
y_C = tf.placeholder(tf.float32, [None, stacks])
y_M = tf.placeholder(tf.float32, [None,1])
in_amout = tf.placeholder(tf.float32, [None, stacks])
in_value = tf.placeholder(tf.float32, [None, stacks])
in_speed = tf.placeholder(tf.float32, [None, stacks])
in_shoot = tf.placeholder(tf.float32, [None, stacks])
in_health = tf.placeholder(tf.float32, [None, stacks])
in_fly = tf.placeholder(tf.float32, [None, stacks])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
is_train = tf.placeholder(tf.bool)
is_not_test = tf.placeholder(tf.bool)
fcn1 = 20
#fcn2 = 160
#fcn3 = 160
# Store layers weight & bias
weights = {
    'fc1': tf.Variable(tf.random_normal([n_all, fcn1])),
    'fc2': tf.Variable(tf.random_normal([fcn1, fcn1])),
    'fc3': tf.Variable(tf.random_normal([fcn1, fcn1])),
    'casul': tf.Variable(tf.random_normal([fcn1, stacks])),
    'mana': tf.Variable(tf.random_normal([fcn1, 1]))
}

biases = {
    'fc1': tf.Variable(tf.random_normal([fcn1])),
    'fc2': tf.Variable(tf.random_normal([fcn1])),
    'fc3': tf.Variable(tf.random_normal([fcn1])),
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
    fc1 = tf.matmul(x, weights['fc1'])+biases['fc1']
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.layers.batch_normalization(fc1, training=is_train)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # fc1 = tf.matmul(fc1, weights['fc2']) + biases['fc2']
    # fc1 = tf.nn.relu(fc1)
    # fc1 = tf.layers.batch_normalization(fc1, training=is_train)
    #fc1 = tf.matmul(fc1, weights['fc3']) + biases['fc3']
    # fc1 = tf.layers.batch_normalization(fc1, training=is_train)
    #fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    casul = tf.matmul(fc1, weights['casul'])+biases['casul']
    casul = tf.layers.batch_normalization(casul, training=is_train)
    casul = tf.nn.sigmoid(casul)
    mana = tf.matmul(fc1, weights['mana']) + biases['mana']
    mana = tf.layers.batch_normalization(mana, training=is_train)
    mana = tf.nn.sigmoid(mana)
    return casul,mana

predC,predM = conv_net(x,keep_prob)
#calsu = predC * in_amout *1000
#cm = predM * in_M
y_C = utils.smart_cond(is_not_test,lambda : y_C * 1000,lambda : y_C)
calsu = utils.smart_cond(is_not_test,lambda :(predC * in_amout * 1000),lambda:(tf.floor(predC*in_amout)))#tf.round(predC * in_amout)
cm = utils.smart_cond(is_not_test,lambda :(predM * in_M ),lambda:(tf.floor(predM * in_M)))
lossC = tf.abs((tf.reduce_sum(calsu *in_value,1) - tf.reduce_sum(y_C*in_value,1)))#每个slot比例*总数取整 再*aiValue
lossCN = tf.abs(tf.reduce_sum(calsu,1) - tf.reduce_sum(y_C, 1))
lossFly = tf.abs((tf.reduce_sum(calsu *in_fly,1) - tf.reduce_sum(y_C*in_fly,1)))
lossShoot = tf.abs((tf.reduce_sum(calsu * in_shoot, 1) - tf.reduce_sum(y_C * in_shoot, 1)))
lossSpeed = tf.abs((tf.reduce_sum(calsu * in_speed, 1) - tf.reduce_sum(y_C * in_speed, 1)))
lossHealth = tf.abs((tf.reduce_sum(calsu * in_health, 1) - tf.reduce_sum(y_C * in_health, 1)))
lossM = tf.abs((cm - y_M))
accuracyC = tf.reduce_mean((lossC))
accuracyCN = tf.reduce_mean((lossCN))
accuracyFly = tf.reduce_mean((lossFly))
accuracyShoot = tf.reduce_mean((lossShoot))
accuracySpeed = tf.reduce_mean((lossSpeed))
accuracyHealth = tf.reduce_mean((lossHealth))
accuracyM = tf.reduce_mean((lossM))
cost = (accuracyCN+accuracyFly+accuracyShoot+accuracySpeed+accuracyHealth+accuracyM)/6
current_epoch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05,
                                           current_epoch,
                                           decay_steps=training_iters,
                                           decay_rate=0.03)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=current_epoch)

def vcnn(train,path,saveModelPath):

    bx, bxm, byc, bym, b_amount, b_value, origPlane,add = lj.loadData(path)
    b_fly = origPlane[:, 0, :, 2]
    b_shoot = origPlane[:, 0, :, 3]
    b_speed = origPlane[:, 0, :, 4]
    b_health = origPlane[:, 0, :, 5]
    # Construct model
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Launch the graph
    if train:
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            min_err = 65535
            while step  < training_iters:
                # Run optimization op (backprop)
                current_epoch = step
                sess.run([optimizer], feed_dict={x: bx,
                                                 y_C: byc,
                                                 y_M:bym,
                                                 in_amout:b_amount,
                                                 in_value:b_value,
                                                 in_M:bxm,
                                                 in_fly:b_fly,
                                                 in_shoot:b_shoot,
                                                 in_speed:b_speed,
                                                 in_health:b_health,
                                               keep_prob: dropout,is_train:True,is_not_test:True})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    accC,accCN,accM = sess.run([accuracyC,accuracyCN, accuracyM], feed_dict={x: bx,
                                                 y_C: byc,
                                                 y_M:bym,
                                                 in_amout:b_amount,
                                                 in_value:b_value,
                                                 in_M:bxm,
                                                 in_fly:b_fly,
                                                 in_shoot:b_shoot,
                                                 in_speed:b_speed,
                                                 in_health:b_health,
                                               keep_prob: 1,is_train:False,is_not_test:True})
                    if accCN < min_err:
                        min_err = accCN
                        saver.save(sess, saveModelPath,global_step=step)
                    print("Iter " + str(step) + ", errorC= " + \
                          "{:.6f}".format(accC) + ", errorCN= " + \
                          "{:.6f}".format(accCN) + ", errorM= " + \
                          "{:.6f}".format(accM))
                step += 1
            print("训练结束")

            saver.save(sess, saveModelPath)
            sess.close()
    else:
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, saveModelPath)
            cas, mc, lsC = sess.run([calsu, cm, lossC], feed_dict={x: bx,
                                                 y_C: byc,
                                                 y_M:bym,
                                                 in_amout:b_amount,
                                                 in_value:b_value,
                                                 in_M:bxm,
                                                 in_fly:b_fly,
                                                 in_shoot:b_shoot,
                                                 in_speed:b_speed,
                                                 in_health:b_health,
                                                keep_prob: 1, is_train: False,is_not_test: True})

            index = np.argsort(lsC)
            for n in (index):
                print("iter: ", n, "lossC: ", lsC[n])
                print(byc[n], bym[n])
                print((cas[n]), (mc[n]))
                print(np.floor(b_value[n]))
            sess.close()
            return
if __name__ == '__main__':
    #vcnn(True,'./train/', './result/model.ckpt')
    vcnn(False, './train/', './result/model.ckpt')

