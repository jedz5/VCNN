import gym
import numpy as np
from collections import defaultdict
from tianshou.data import Batch,ReplayBuffer

# m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# n = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
#
# print('list(zip(m,n)):\n', list(zip(m, n)))
# print("*zip(m, n):\n", *zip(m, n))
# print("*zip(*zip(m, n)):\n", *zip(*zip(m, n)))
#
# m2, n2 = zip(*zip(m, n))
# print(m == list(m2) and n == list(n2))


# a = np.array([0,0,0,1,0,0,1])
# s = a.sum()
# lk = np.array(range(s-1,-1,-1))
# a[a == 1] += lk


# a = Batch(rew = [1,2,3,4])
# b = Batch(rew = [3,4,5,6],act = [11,21,31,41])
# c = Batch.cat([a[:2],Batch(b)])
# b.rew[0] = 111
# print(np.random.binomial(1,0.2))
global_buffer = ReplayBuffer(500,ignore_obs_next=True)
global_buffer.add(obs=1, act=2, rew=1, done=1)
global_buffer.add(obs=1, act=2, rew=1, done=2)
a = global_buffer.sample(0)[0]
b = Batch(rew = [3,4,5,6],act = [11,21,31,41])
c = Batch.cat([a,b])