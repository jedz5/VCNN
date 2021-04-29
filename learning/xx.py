import gym
import numpy as np
import torch
from collections import defaultdict
from tianshou.data import Batch,ReplayBuffer
np.set_printoptions(precision=2,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.2f}'.format})
# m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# n = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
#
# print('list(zip(m,n)):\n', list(zip(m, n)))
# print("*zip(m, n):\n", *zip(m, n))
# print("*zip(*zip(m, n)):\n", *zip(*zip(m, n)))
#
# m2, n2 = zip(*zip(m, n))
# print(m == list(m2) and n == list(n2))

# buf = ReplayBuffer(10,ignore_obs_next=True)
# a = Batch(obs=1,act=2,rew=0,done=0)
# buf.add(a)
# buf.add(a)
# b = Batch(obs=1,act=2,rew=1,done=1)
# buf.add(b)
# batch1= buf.sample(0)[0]
# start = buf._index
# buf.add(a)
# buf.add(a)
# buf.add(b)
# buf.add(a)
# buf.add(a)
# buf.add(b)
# end = buf._index
# # cur = buf.rew[list(range(start,len(buf)))+list(range(end))]
# # cur = buf.rew[list(range(start,end))]
# cur = buf.rew[start:end]
# def cumulate_reward(batch):
#     a = batch
#     s = int(a.sum())
#     lk = np.array(range(s - 1, -1, -1))
#     a[a > 0.5] += lk
# cumulate_reward(cur)

a = {"abc":1}
a["c"] = 1
