import json
from collections import defaultdict


import pytest
import torch
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy
from ding.worker import EpisodeReplayBuffer
from dizoo.classic_control.bitflip.config import bitflip_her_dqn_config
from easydict import EasyDict
from tianshou.data import Batch
from torch.optim import SGD
import numpy as np

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
def fff(*par,e=0):
    print(e)
    ddd(*par)
def param_test():
    d = 1
    # bcd = {'b':1,'c':2,'a':3}
    # abc(d=d,**bcd)
    fff(2,3,4,5)
class Ax:
    def __init__(self,aa):
        self.aa = aa
    def __hash__(self):
        return hash(self.aa)

    def __eq__(self, other):
        return self.aa == other.aa
def hash_test():
    dic = {}
    dic[Ax(3)] = 3
    dic[Ax(4)] = 4
    dic[Ax(4)] = 5
    print(dic.keys())
def xx():
    with open(r"D:\project\VCNN\ENV\battles\0.json") as JsonFile:
        js = json.load(JsonFile)
    curr = js["stacks"]
    attri_stack_orig = np.array(curr)
    attri_stack = np.copy(attri_stack_orig)
    r'''部队基础总生命值'''
    health_baseline = attri_stack[..., 4] * attri_stack[..., 6]
    health_baseline = health_baseline.astype('int')
    r'''部队基础总生命值中最大值'''
    health_baseline_max = np.max(health_baseline, axis=-1)
    r'''部队当前总生命值'''
    health_current = np.clip(attri_stack[..., 3] - 1, 0, np.inf) * attri_stack[..., 6] + attri_stack[..., 5]
    health_current = health_current.astype('int')
    r'''部队基础总生命值 / 部队基础总生命值中最大值'''
    health_ratio_bymax = health_baseline * 10 // health_baseline_max[..., None]
    r'''部队当前总生命值 / 部队基础总生命值'''
    health_current_ratio = (health_current * 10 // (health_baseline + 1E-9)).astype('int') + (
            health_current > 0).astype('int')
    r'''部队当前数量 / 部队基础数量'''
    amount_ratio = (attri_stack[..., 3] * 10 // (attri_stack[..., 4] + 1E-9)).astype('int') + (
            attri_stack[..., 3] > 0).astype('int')
    r'''远程弹药数量 / 基数16'''
    shoots_ratio = (attri_stack[..., 14] * 4 // 16.0001).astype('int') + (attri_stack[..., 14] > 0).astype('int')
    attri_stack[..., 3] = amount_ratio
    attri_stack[..., 5] = health_current_ratio
    attri_stack[..., 6] = health_ratio_bymax
    attri_stack[..., 14] = shoots_ratio
    print()
def list_sort():
    l = [Ax(6), Ax(3), Ax(2), Ax(5), Ax(4)]
    l.sort(key=lambda x: x.aa)
    print(l)
def epi_buffer():
    cfg = EasyDict(replay_buffer_size=50, deepcopy=False, exp_name='test_episode_buffer', enable_track_used_data=False)
    replay_buffer = EpisodeReplayBuffer(
        cfg, exp_name=cfg.exp_name, instance_name='episode_buffer'
    )
    for i in range(1, 10):
        replay_buffer.push(Batch.stack([Batch({'a': 1, 'b': 2})] * i), i)
    print()
M = 5
if __name__ == '__main__':
    param_test()