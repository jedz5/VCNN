import gym
import numpy as np
import torch
from collections import defaultdict
from tianshou.data import Batch,ReplayBuffer
np.set_printoptions(precision=2,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.2f}'.format})



# batch_rew = np.array([1,0,0,0.1,0,0.2,0,0,0.3,0,-4])
# start = 1
# br_index = np.where(batch_rew[start:] > 0)
# end_bias = br_index[-1][-1] + 1
# print(batch_rew[start:start+end_bias])
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

# A = np.array([[1,1,1]]).transpose()
# a = np.linalg.pinv(A)


# pi = np.array([[0.2,0.8,0,0,0,0], # a1/s1 a2/s1
#                [0, 0, 0.3, 0.7, 0, 0], # a1/s2 a2/s2
#                [0, 0, 0, 0,0.6, 0.4]]) # a1/s3 a2/s3
# pt = np.array([[.3,.4,.3],   #s1/s1a1 s2/s1a1 s3/s1a1
#                [.2, .5, .3], #s1/s1a2 s2/s1a2 s3/s1a2
#                [.3, .2, .5], #s1/s2a1 s2/s2a1 s3/s2a1
#                [.13, .4, .47], #s1/s2a2 s2/s2a2 s3/s2a2
#                [.3, .48, .22], #s1/s3a1 s2/s3a1 s3/s3a1
#                [.1, .1, .8] #s1/s3a2 s2/s3a2 s3/s3a2
#                ])
# pss = np.matmul(pi,pt)
# tmp = np.eye(3) - 0.9*pss
# c = np.linalg.inv(tmp)
# print(c)
ll = [1, 2, 5, 2, 2,3]
it = iter(ll)
i = 0
while True:
    x = next(it, 'a')
    if x == 2:
        ll.pop(i)
    else:
        i += 1
    print(x)
    if x == 'a':
        break