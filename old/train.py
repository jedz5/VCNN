# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import deque
from ENV import H3_battle as bat
from old.mcts_alpha import MCTSPlayer
import traceback
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from old.policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from ENV.H3_battle import logger
class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        #self.board_width = 6
        #self.board_height = 6
        #self.n_in_row = 4
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 256  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 256  # mini-batch size for training
        self.recent_sample_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5000  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 512
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.tmp_battle = bat.Battle(load_file="./train/selfplay1.json")
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(bat.Battle.bFieldWidth - 2, bat.Battle.bFieldHeight,
                                                   bat.Battle.bFieldStackPlanes, bat.Battle.bTotalFieldSize,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(bat.Battle.bFieldWidth - 2, bat.Battle.bFieldHeight,
                                                   bat.Battle.bFieldStackPlanes, bat.Battle.bTotalFieldSize)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1,battle=self.tmp_battle)

    def collect_selfplay_data(self, n_games=1,take_control=0,init_model = 0):
        """collect self-play data for training"""
        if take_control:
            self.mcts_player = bat.BPlayer()
        for i in range(n_games):
            self.tmp_battle = bat.Battle(load_file="./train/selfplay1.json")
            if take_control:
                play_data = np.load('play_data.npy')
            else:
                play_data = self.tmp_battle.start_self_play(self.mcts_player,take_control,temp=1.0)
                play_data = list(play_data)[:]
                #np.save('play_data.npy',play_data)
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)


    def policy_update(self):
        """update the policy-value net"""
        if len(self.data_buffer) < self.batch_size:
            mini_batch = self.data_buffer
        else:
            mini_batch = random.sample(list(self.data_buffer)[-self.recent_sample_size:], self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        side_batch = [data[2] for data in mini_batch]
        left_batch = [data[3] for data in mini_batch]
        left_base_batch = [data[4] for data in mini_batch]
        right_batch = [data[5] for data in mini_batch]
        right_base_batch = [data[6] for data in mini_batch]
        old_probs, valueL,valueR, fvalue_left, fvalue_right = self.policy_value_net.policy_value(state_batch)
        logger.info("old_valueL = {}".format(valueL))
        logger.info("old_valueR = {}".format(valueR))
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    side_batch,left_batch,left_base_batch,right_batch,right_base_batch,
                    self.learn_rate*self.lr_multiplier)
            if i%100 == 0:
                logger.info(("i:{},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               #"explained_var_old:{:.3f},"
               #"explained_var_new:{:.3f}"
               ).format(i,
                        self.lr_multiplier,
                        loss,
                        entropy))
            new_probs, new_vL,new_vR,fvalue_left, fvalue_right = self.policy_value_net.policy_value(state_batch)
            #side_batch_tmp = [x[0] for x in side_batch]
            #computedVaue = side_batch_tmp*((fvalue_left*left_batch).sum(axis=1)/(fvalue_left*left_base_batch).sum(axis=1) - (fvalue_right*right_batch).sum(axis=1)/(fvalue_right*right_base_batch).sum(axis=1))
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if loss < 0.01:
                break
            # if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
            #     break
        # adaptively adjust the learning rate
        logger.info("new_valueL = {}".format(new_vL))
        logger.info("new_valueR = {}".format(new_vR))
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        logger.info(("i:{}, kl:{:.5f},"
                         "lr_multiplier:{:.3f},"
                         "loss:{},"
                         "entropy:{},"
                         # "explained_var_old:{:.3f},"
                         # "explained_var_new:{:.3f}"
                         ).format(i, kl,
                                  self.lr_multiplier,
                                  loss,
                                  entropy))
        # explained_var_old = (1 -
        #                      np.var(np.array(winner_batch) - old_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(winner_batch) - new_v.flatten()) /
        #                      np.var(np.array(winner_batch)))

                        # explained_var_old,
                        # explained_var_new))
        return loss, entropy

    # def policy_evaluate(self, n_games=10):
    #     """
    #     Evaluate the trained policy by playing against the pure MCTS player
    #     Note: this is only for monitoring the progress of training
    #     """
    #     current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
    #                                      c_puct=self.c_puct,
    #                                      n_playout=self.n_playout)
    #     pure_mcts_player = bat.Battle.BPlayer()
    #     win_cnt = defaultdict(int)
    #     for i in range(n_games):
    #         winner = self.game.start_play(current_mcts_player,
    #                                       pure_mcts_player,
    #                                       start_player=i % 2,
    #                                       is_shown=0)
    #         win_cnt[winner] += 1
    #     win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    #     log("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
    #             self.pure_mcts_playout_num,
    #             win_cnt[1], win_cnt[2], win_cnt[-1]))
    #     return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size,0)
                logger.info("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if 1: #len(self.data_buffer) > self.batch_size
                    loss, entropy = self.policy_update()
                    logger.info("selfplay epoch= {} loss = {},entropy ={}".format(i,loss,entropy))
                # check the performance of the current model,
                # and save the model params
                    self.policy_value_net.save_model('./model/current_policy.model')
                # if (i+1) % self.check_freq == 0:
                #     logger.info("current self-play batch: {}".format(i+1))
                #     win_ratio = self.policy_evaluate()
                #     self.policy_value_net.save_model('./current_policy.model')
                #     if win_ratio > self.best_win_ratio:
                #         logger.info("New best policy!!!!!!!!")
                #         self.best_win_ratio = win_ratio
                #         # update the best_policy
                #         self.policy_value_net.save_model('./best_policy.model')
                #         if (self.best_win_ratio == 1.0 and
                #                 self.pure_mcts_playout_num < 5000):
                #             self.pure_mcts_playout_num += 1000
                #             self.best_win_ratio = 0.0
        except Exception as e:
            logger.error(e)
            traceback.print_exc()


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
