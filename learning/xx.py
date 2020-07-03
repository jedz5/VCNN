import torch
import time
import numpy as np
from tianshou.data import ReplayBuffer
# a = np.array([1,1,1,1,1,1,1])
# np.save("d:/a.npy",a)
# buffer_def = ReplayBuffer(100, ignore_obs_next=True)
# buffer_att = ReplayBuffer(100, ignore_obs_next=True)
# buffer_def.add(obs=1, act=1, rew=1, done=1)
# # buffer_att.update(buffer_def)
# buffer_def.update(buffer_att)

a = np.array([1,1,1,0,0,0])
c = np.arange(len(a))
b = np.random.choice(c,p=(a/a.sum()))