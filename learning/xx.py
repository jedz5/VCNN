from collections import defaultdict


import pytest
import torch
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy
from torch.optim import SGD
import torch.nn.functional as F

def test_upgo2():
    S, B, A, N2 = 4, 8, 5, 7

    reward = torch.zeros(S, B,dtype=torch.float32)
    reward[-1,:2] = 1.
    reward[-1, 2:] = -1.
    bootstrap_values_orig = torch.zeros((S+1,),dtype=torch.float32).requires_grad_(True)

    # upgo loss
    logit_q = torch.zeros((S, A),dtype=torch.float32,requires_grad=True)
    opt = SGD([logit_q,bootstrap_values_orig],0.1)
    action = torch.zeros(S,B,dtype=torch.long)
    for i in range(10):
        logit = torch.stack([logit_q.softmax(-1)] * B).transpose(0,1)
        bootstrap_values = torch.stack([bootstrap_values_orig] * B).transpose(0,1)
        rhos = torch.ones(S, B)
        up_loss = upgo_loss(logit, rhos, action, reward, bootstrap_values)
        with torch.no_grad():
            returns = upgo_returns(reward, bootstrap_values)

        print("values:")
        print(bootstrap_values_orig.detach().numpy())
        mask = reward + bootstrap_values[1:] - bootstrap_values[:-1]
        print("mask:")
        print(mask.transpose(0,1).detach().numpy())
        print("returns:")
        print(returns.transpose(0,1).detach().numpy())
        v_loss = 1/2 *(returns - bootstrap_values[:-1]) ** 2
        v_loss = v_loss.mean()
        # assert logit.requires_grad
        # assert bootstrap_values.requires_grad
        # for t in [logit, bootstrap_values]:
        #     assert t.grad is None
        opt.zero_grad()
        loss = up_loss + v_loss
        loss.backward()
        opt.step()
def defaultdict_int():
    return defaultdict(int)
def xx():
    # tensor_0 = torch.arange(3, 12).view(3, 3)
    # print(tensor_0)
    # index = torch.tensor([2, 1, 0,2,1,1,2])
    # tensor_1 = tensor_0.gather(0,index.unsqueeze(-1).expand((index.shape[0],tensor_0.shape[1])).long())
    # print(tensor_1)
    pass
def upgo_VQG():
    Q = defaultdict(dict)
    V = defaultdict(dict)
    sars_count = defaultdict(defaultdict_int)
    Q[(1, 3)][1] = torch.tensor(-1.)
    Q[(1, 4)][2] = torch.tensor(-.02)
    V[(1, 3)] = max(Q[(1, 3)].values())
    V[(1, 4)] = max(Q[(1, 4)].values())
    #
    sars_count[((1, 1),4)][(1, 4)] = 99
    sars_count[((1, 1),4)][(1, 3)] = 1
    s = (1, 1)
    a = 4
    Q[s][a] = sum([sars_count[(s,a)][s_] * V[s_] for s_ in sars_count[(s,a)].keys()]) / sum(sars_count[(s,a)].values())
    print(Q)
    #(1, 1), 4, (1, 4) ][(1, 4), 5, (1, 5)
    sars_count[((1, 4), 5)][(1, 5)] += 1
    Q[(1, 4)][5] = 1.
    s = (1, 1)
    a = 4
    s_ = (1,4)
    sars_count[s,a][s_] += 1
    V[s_] = max(Q[s_].values())
    Q[s][a] = sum([sars_count[(s, a)][s_] * V[s_] for s_ in sars_count[(s, a)].keys()]) / sum(
        sars_count[(s, a)].values())
    print(Q)
def ddd(a,b,c,d):
    print('hh')
def abc(*par,**par2):
    ddd(*par,**par2)
if __name__ == '__main__':
    a = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.],requires_grad=True)
    opt = SGD([a],lr=0.1)
    # loss = .5 * (a - 1)**2
    loss = .5 * F.mse_loss(a, torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.]), reduction='sum')#.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a)

