import gym
import numpy as np
import torch
from collections import defaultdict
from tianshou.data import Batch,ReplayBuffer
np.set_printoptions(precision=2,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.2f}'.format})

# class A:
#     def hh(self):
#         print("hh")
# class B(A):
#     def __init__(self, hh_func=None):
#         if hh_func:
#             self.hh = self.hh2
#     def hh2(self):
#         hh2()
#     def nn(self):
#         pass
# def hh2():
#     print("??")
# def hh3():
#     print("**")
#
# hh2 = hh3
# b = B()
# b.hh2()
# c = B(hh3)
# c.hh()
# b.hh()
batch_rew = np.array([0,0,0.1,0,0,0.2,0,0,0.3])
# a = np.argwhere(batch_rew> 0).squeeze()
# b = np.append(0,(a + 1)[:-1])
# # a = batch_rew[batch_rew > 0][::-1]
# # batch_rew[batch_rew > 0] = np.add.accumulate(a)[::-1]
#
# buf = ReplayBuffer(10,ignore_obs_next=False)
# a = Batch(obs=1,act=2,rew=0,done=False)
# buf.add(a)
# buf.add(a)
# b = Batch(obs=1,act=2,rew=1,done=True)
# buf.add(b)
# buf.add(a)
# start = buf._index
# buf.add(a)
# buf.add(b)
# buf.add(a)
# buf.add(a)
# buf.add(b)
# end = buf._index
# buf.add(a)
# buf.add(a)
# buf.add(b)
# import PG_model.h3_ppo
# PG_model.h3_ppo.cumulate_reward_2(buf,start,end)