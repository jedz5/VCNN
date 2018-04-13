# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import Battle

#bFieldWidth = 17
#bFieldHeight = 11
class PolicyValueNet():
    def __init__(self, bfield_width, bfield_height,planes,num_actions,model_file=None):
        self.bfield_width = bfield_width
        self.bfield_height = bfield_height
        self.planes = planes
        self.num_actions = num_actions
        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, self.bfield_height, self.bfield_width, self.planes])
        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_states,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * bfield_height * bfield_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=self.num_actions,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * bfield_height * bfield_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_value = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # 5 Evaluation Networks
        self.fValue_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)
        self.fValue_conv_flat = tf.reshape(
            self.fValue_conv, [-1, 2 * bfield_height * bfield_width])
        self.fValue_fc1 = tf.layers.dense(inputs=self.fValue_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.fValue_fc_left = tf.layers.dense(inputs=self.fValue_fc1,
                                              units=7, activation=tf.nn.sigmoid)
        self.fValue_fc_right = tf.layers.dense(inputs=self.fValue_fc1,
                                            units=7, activation=tf.nn.sigmoid)
        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels_left_hp = tf.placeholder(tf.float32, shape=[None, 7])
        self.labels_right_hp = tf.placeholder(tf.float32, shape=[None, 7])
        self.labels_left_hp_0 = tf.placeholder(tf.float32, shape=[None, 7])
        self.labels_right_hp_0 = tf.placeholder(tf.float32, shape=[None, 7])
        self.labels_side = tf.placeholder(tf.float32, shape=[None,1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_value
        # 3-1. Value Loss function
        self.fValueLoss2 = tf.reduce_sum(self.fValue_fc_right * self.labels_right_hp,1,keep_dims=True) / tf.reduce_sum(self.fValue_fc_right * self.labels_right_hp_0,1,keep_dims=True)
        self.fValueLoss1 = tf.reduce_sum(self.fValue_fc_left * self.labels_left_hp,1,keep_dims=True) / tf.reduce_sum(self.fValue_fc_left * self.labels_left_hp_0,1,keep_dims=True)
        self.fValueLoss1_2 = self.labels_side * (self.fValueLoss1 - self.fValueLoss2)
        self.value_loss = tf.losses.mean_squared_error(self.evaluation_value,self.fValueLoss1_2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, self.num_actions])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value,fvalue1,fvalue2 = self.session.run(
                [self.action_fc, self.evaluation_value,self.fValue_fc_left,self.fValue_fc_right],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value,fvalue1,fvalue2

    def policy_value_fn(self, battle):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = battle.curStack.legalMoves()
        current_state = battle.currentStateFeature()
        act_probs, value,fvalue1,fvalue2 = self.policy_value([current_state])
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value,fvalue1[0],fvalue2[0]

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
