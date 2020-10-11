from tianshou.data import ReplayBuffer
from tianshou.data import Batch
import tianshou.policy.base as bs
import PG_model.h3_ppo as hp3
import numpy as np
import scipy.signal
import json
import torch
# buffer = ReplayBuffer(500,ignore_obs_next=True)
# for i in range(5):
#     done = (i % 3) == 0 or i == 4
#     rew = 0
#     if done:
#         rew = 1
#     buffer.add(obs={"i1":i,"i2":i}, act=i, rew=rew, done=done,info={"cc":i},policy={"logp":i,"value":i})
# buffer.done[len(buffer) -1] = True
# data,indice = buffer.sample(0)

# with open("ENV/creatureData.json") as JsonFile:
#     crList = json.load(JsonFile)["creatures"]
#     for i in range(100):
#         x = crList[i]
#         r1 = x['speed'] * int((x['max_damage'] + x['min_damage']) / 2) * x['attack'] * (1 + int(x['shoot']))
#         r2 = x['health'] * x['defense']bbb
#         print(f"{x['name']} = {r1} + {r2} - {x['ai_value']} = {r1 + r2 - x['ai_value']}")
#         print(f"{x['name']} = {r1} + {r2} - {x['fight_value']} = {r1 + r2 - x['fight_value']}")
x = torch.arange(-100,100)

a = torch.tensor(0.05,requires_grad=True)
b = torch.tensor(-0.03,requires_grad=True)
c = torch.tensor(-0.02,requires_grad=True)
opt = torch.optim.Adam([a,b,c],lr=1)
for i in range(100):
    xx = torch.tensor(np.random.choice(x,32))
    yy = 3 * xx * xx + 8 * xx + 5
    opt.zero_grad()
    loss = a*xx*xx + b*xx + c - yy
    loss = loss * loss
    loss = loss.mean()
    loss.backward()
    opt.step()
    print(f"{a}  {b}  {c}")