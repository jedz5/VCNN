import torch
import time
import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.data import Batch

# data = Batch(a=np.array([[0.0, 2.0], [1.0, 3.0]]), b=[[5, -5]])
# data = ReplayBuffer(20,0,ignore_obs_next=True)
# for i in range(10):
#     done = 1 if i % 3 == 0 else 0
#     data.add(i,i,i,done)
#
# index = np.arange(10)
# print(data[index])
# print(data[index])
# print(data[index])

a = np.zeros([2,3,4,5])
b = a[0, ..., 0]