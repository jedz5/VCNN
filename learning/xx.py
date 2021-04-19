import gym
import numpy as np
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


# a = np.array([0,0,0,1,0,0,1])
# s = a.sum()
# lk = np.array(range(s-1,-1,-1))
# a[a == 1] += lk


a = np.array(range(140)) / 140
print(a)