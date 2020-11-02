import numpy as np
import torch
from learning.mcts_q import state_node_determ
# l = []
# for i in range(10):
#     l.append(state_node_determ(i,0,0,0,0))
#
# # a = np.argmax(l,)
# a = max(l,key=lambda x:x.s)
from learning.mcts_q import snake_net
a = torch.randint(0,36,(100,16))
net = snake_net(36)
b = net(a)
print()