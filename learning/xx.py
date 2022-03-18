import copy
import json
import logging
from collections import defaultdict
import re
from collections.abc import Sequence, Mapping
# from matplotlib.pyplot import hist, ylabel, xlabel
# from tianshou.data import Batch
from typing import List, Dict, Union, Any
from torch._six import string_classes
import collections.abc as container_abcs
from ding.compatibility import torch_gt_131

from ding_model.max_tree_collector import MaxTreeCollector

int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')

# import pytest
import torch
# from ding.utils.data.collate_fn import default_collate
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy
from ding.worker import EpisodeReplayBuffer
# from dizoo.classic_control.bitflip.config import bitflip_her_dqn_config
# from easydict import EasyDict
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
class Qvalue:
    def __init__(self,q=0.):
        self.v = q
    def __repr__(self):
        return f"{self.v}"
    def __deepcopy__(self,memo):
        raise Exception('deepcopy not allowed')
    def __copy__(self):
        raise Exception('copy not allowed')
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

def defaultdict_value():
    return defaultdict(Qvalue)
def defaultdict_value_test():
    a = defaultdict(defaultdict_value)
    a[(1,2,3)][1].q = 1.
    print(a)

def param_test():
    def ddd(a, b, c, d):
        print('hh')

    def abc(*par, **par2):
        ddd(*par, **par2)

    def fff(*par, e=0):
        print(e)
        ddd(*par)
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
def h3_obs_norm():
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

import numpy as np
# from treevalue import FastTreeValue

# T, B = 3, 4
# power = FastTreeValue.func()(np.power)
# stack = FastTreeValue.func(subside=True)(np.stack)
# split = FastTreeValue.func(rise=True)(np.split)
def without_treevalue(batch_):
    mean_b_list = []
    even_index_a_list = []
    for i in range(len(batch_)):
        for k, v in batch_[i].items():
            if k == 'a':
                v = v.astype(np.float32)
                even_index_a_list.append(v[::2])
            elif k == 'b':
                v = v.astype(np.float32)
                transformed_v = np.power(v, 2) + 1.0
                mean_b_list.append(transformed_v.mean())
            elif k == 'c':
                for k1, v1 in v.items():
                    if k1 == 'd':
                        v1 = v1.astype(np.float32)
                    else:
                        print('ignore keys: {}'.format(k1))
            else:
                print('ignore keys: {}'.format(k))
    for i in range(len(batch_)):
        even_index_a = np.stack(even_index_a_list, axis=0)
    return batch_, mean_b, even_index_a
def with_treevalue(batch_):
    batch_ = [FastTreeValue(b) for b in batch_]
    batch_ = stack(batch_)
    batch_ = batch_.astype(np.float32)
    batch_.b = power(batch_.b, 2) + 1.0
    batch_.c.noise = np.random.random(size=(B, 3, 4, 5))
    mean_b = batch_.b.mean()
    even_index_a = batch_.a[:, ::2]
    batch_ = split(batch_, indices_or_sections=B, axis=0)
    return batch_, mean_b, even_index_a

def tree_func_test():
    def get_data():
        return {
            'a': np.random.random(size=(T, 8)),
            'b': np.random.random(size=(6,)),
            'c': {
                'd': np.random.randint(0, 10, size=(1,))
            }
        }
    a = [get_data() for i in range(10)]
    # batch_, mean_b, even_index_a = without_treevalue(a)
    batch_, mean_b, even_index_a = with_treevalue(a)
    print()

def compare_sort():
    def compare11(x,y):
        return x >= y

    def cmp_to_key11(mycmp):
        """Convert a cmp= function into a key= function"""

        class K(object):
            __slots__ = ['obj']

            def __init__(self, obj):
                self.obj = obj

            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0

            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0

            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0

            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0

            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0

            __hash__ = None

        return K
    ll = [1, 3, 2, 5, 9, 4, 2]
    ll.sort(key=cmp_to_key11(compare11))
def tree_tensor_test():
    import builtins
    import os
    from functools import partial

    import treetensor.torch as torch

    print = partial(builtins.print, sep=os.linesep)
    # create a tree tensor
    t = torch.randn({'a': (2, 3), 'b': {'x': (3, 4)}})
    print(t)
    print(torch.randn(4, 5))  # create a normal tensor
    print()

    # structure of tree
    print('Structure of tree')
    print('t.a:', t.a)  # t.a is a native tensor
    print('t.b:', t.b)  # t.b is a tree tensor
    print('t.b.x', t.b.x)  # t.b.x is a native tensor
    print()

    # math calculations
    print('Math calculation')
    print('t ** 2:', t ** 2)
    print('torch.sin(t).cos()', torch.sin(t).cos())
    print()

    # backward calculation
    print('Backward calculation')
    t.requires_grad_(True)
    t.std().arctan().backward()
    print('grad of t:', t.grad)
    print()

    # native operation
    # all the ops can be used as the original usage of `torch`
    print('Native operation')
    print('torch.sin(t.a)', torch.sin(t.a))  # sin of native tensor

def tree_build():
    import os
    from treevalue import FastTreeValue
    data = [FastTreeValue({'state':{'global': np.ones((4,4)) + i , 'local': np.ones((4,4)) +i},'action':np.random.randint(0,5),'reward':i}) for i in range(10)]
    last = True
    for t in reversed(data):
        if last:
            last = False
        else:
            t.next_obs = last_obs
        last_obs = t
    # t.a = FastTreeValue({'s': {'global': np.ones((5,5)), 'local': np.ones((5,5))}})
    # t = FastTreeValue({'a': 1, (2,2,2): 2, 'x': {'c': 3, 'd': 4}})
    print("Original tree:", t, sep=os.linesep)


    # Get values
    print("Value of t.a: ", t.a)
    print("Value of t.x.c:", t.x.c)
    print("Value of t.x:", t.x, sep=os.linesep)

    # Set values
    t.a = 233
    print("Value after t.a = 233:", t, sep=os.linesep)
    t.x.d = -1
    print("Value after t.x.d = -1:", t, sep=os.linesep)
    t.x = FastTreeValue({'e': 5, 'f': 6})
    print("Value after t.x = FastTreeValue({'e': 5, 'f': 6}):", t, sep=os.linesep)
    t.x.g = {'e': 5, 'f': 6}
    print("Value after t.x.g = {'e': 5, 'f': 6}:", t, sep=os.linesep)

    # Delete values
    del t.x.g
    print("Value after del t.x.g:", t, sep=os.linesep)

def test_EpisodeSerialCollector():
    from ding.worker import EpisodeSerialCollector
    from ding.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
    from ding.policy import DQNPolicy
    from ding.model import DQN
    from dizoo.classic_control.cartpole.envs import CartPoleEnv
    env = BaseEnvManager([lambda: CartPoleEnv({}) for _ in range(8)], BaseEnvManager.default_config())
    env.seed(0)
    model = DQN(obs_shape=4, action_shape=1)
    policy = DQNPolicy(DQNPolicy.default_config(), model=model).collect_mode
    collector = EpisodeSerialCollector(EpisodeSerialCollector.default_config(), env, policy)

    collected_episode = collector.collect(
        n_episode=18, train_iter=collector._collect_print_freq, policy_kwargs={'eps': 0.5}
    )
    assert len(collected_episode) == 18
    assert all([e[-1]['done'] for e in collected_episode])
    assert all([len(c) == 0 for c in collector._traj_buffer.values()])
def traj_q():
    def default_collate(batch: Sequence,
                        cat_1dim: bool = True,
                        ignore_prefix: list = ['collate_ignore']) -> Union[torch.Tensor, Mapping, Sequence]:
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch_gt_131() and torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, directly concatenate into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            if elem.shape == (1,) and cat_1dim:
                # reshape (B, 1) -> (B)
                return torch.cat(batch, 0, out=out)
                # return torch.stack(batch, 0, out=out)
            else:
                return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return default_collate([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim)
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, Qvalue):
            return torch.tensor([x.q for x in batch], dtype=torch.float32)
        elif isinstance(elem, int_classes):
            dtype = torch.bool if isinstance(elem, bool) else torch.int64
            return torch.tensor(batch, dtype=dtype)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            ret = {}
            for key in elem:
                if any([key.startswith(t) for t in ignore_prefix]):
                    ret[key] = [d[key] for d in batch]
                else:
                    ret[key] = default_collate([d[key] for d in batch], cat_1dim=cat_1dim)
            return ret
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate(samples, cat_1dim=cat_1dim) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples, cat_1dim=cat_1dim) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
    htable = {}
    htable[(1, 2, 3)] = Qvalue(1.)
    htable[(2, 2, 3)] = Qvalue(1.1)
    htable[(3, 2, 3)] = Qvalue(1.2)
    traj = [Batch({'obs': (1, 2, 3), 'q': htable[(1, 2, 3)]}), Batch({'obs': (2, 2, 3), 'q': htable[(2, 2, 3)]}),
            Batch({'obs': (3, 2, 3), 'q': htable[(3, 2, 3)]})]
    traj = Batch.stack(traj)
    # traj = [{'obs': (1, 2, 3), 'q': htable[(1, 2, 3)]}, {'obs': (2, 2, 3), 'q': htable[(2, 2, 3)]},
    #         {'obs': (3, 2, 3), 'q': htable[(3, 2, 3)]}]
    traj2 = traj[:2]
    # del traj
    batch = [traj,traj2]
    htable[(1, 2, 3)].v = 1.3
    for t in batch:
        for i in t:
            if i.q.v < 1.2:
                print(i)
                i.q.v = .5
                print(i)

    batch = default_collate(traj)
    print()
def exp_avg():
    a = [3, 4, 5]  # 1,2,
    lamdda = 0.5
    disc = 1.
    summ = 0.
    for item in a:
        summ += item * disc
        disc *= lamdda
    print(summ)
# def _get_train_sample( data: list):
#     end_reward = data[-1]['reward']
#     last_rew = 0
#     if end_reward < -1:
#         last_done = data[int(-end_reward)]
#         assert data[last_done]
#         data = data[:last_done+1]
#     end_reward = data[-1]['reward']
#     for step in data:
#         step['g'] = end_reward - last_rew
#         step['adv'] = end_reward - last_rew - step['value']
#         last_rew = step['reward']
#     return data
def process_standard_g(data: list,logger=None):
    r"""
    reward = [2,3,4,5]
    prev_reward = reward[:-1] = [0,2,3,4]
    r = reward - prev_reward = [2,1,1,1]
    g = reward[-1] - prev_reward = [5,3,2,1]
    """
    # last_done = data[-1]['real_done']
    # last_done = int(last_done)
    # if last_done < len(data) - 1:
    #     assert data[last_done]['real_done']
    #     data = data[:last_done + 1]
    #     data[-1]['done'] = True
    #     data[-1]['real_done'] = last_done
    #     logger.info(f"end_reward={data[-1]['reward']}")
    win = False
    prev_data = [None] + data[:-1]
    end_reward = 0
    for step, prev_step in zip(data[::-1], prev_data[::-1]):
        # assert step['obs']['action_mask'][step['action']].item() == 1,f"action id = {step['action'].item()}"
        if step["real_done"]:
            if win:
                print(f"end_reward={data[-1]['reward']}")
                end_reward = max(end_reward,0) #最后一场的负值不能传递到前面场次
                print(f"current_reward={end_reward}")
            end_reward += step['reward']
            win = True
        if prev_step:
            step['g'] = end_reward - prev_step['reward'] * (1 - int(prev_step["real_done"]))
        else:
            step['g'] = end_reward
        step['adv'] = step['g'] - step['value']
    return data
def process_maxtree_g(data:list):
    last_start = 0
    reward_cum = 0
    for i,step in enumerate(data):
        if step['real_done']:
            if i+1 < len(data):
                if data[last_start]['obs'] == data[i+1]['obs']:
                    reward_cum += step['reward']
                    step['reward'] = 0
                else:
                    last_start = i+1
                    step['reward'] += reward_cum
                    reward_cum = 0
                    step['done'] = True
            else:
                step['reward'] += reward_cum
    reward_cum = 0
    for step in reversed(data):
        if step['done']:
            step['reward'] += reward_cum
            reward_cum = step['reward']
def test_get_train_sample():
    data = [{'obs':1,'reward':1,'value':0.2,'real_done':False},{'obs':2,'reward':2,'value':0.3,'real_done':False},{'obs':3,'reward':3,'value':0.4,'real_done':False},{'obs':4,'reward':4,'value':0.5,'real_done':False},
            {'obs':5,'reward':5,'value':0.6,'real_done':True},{'obs':1,'reward':1,'value':0.2,'real_done':False},{'obs':1,'reward':2,'value':0.3,'real_done':False},{'obs':1,'reward':3,'value':0.4,'real_done':False},
            {'obs':1,'reward':4,'value':0.5,'real_done':False},
            {'obs':5,'reward':5,'value':0.6,'real_done':True},{'obs':2,'reward':1,'value':0.2,'real_done':False},{'obs':1,'reward':2,'value':0.3,'real_done':False},{'obs':1,'reward':3,'value':0.4,'real_done':False},
            {'obs':1,'reward':4,'value':0.5,'real_done':False},{'obs':1,'reward':3,'value':0.6,'real_done':True},{'obs':2,'reward':1,'value':0.2,'real_done':False},{'obs':1,'reward':2,'value':0.3,'real_done':False},
            {'obs':1,'reward':3,'value':0.4,'real_done':False},
            {'obs':1,'reward':4,'value':0.5,'real_done':False},{'obs':1,'reward':3,'value':0.6,'real_done':True}]
    data[-1]['real_done'] = len(data)-1
    for step in data:
        step['done'] = False
    data[-1]['done'] = True
    data2 = process_maxtree_g(data)
    print(data)


class hash_traj(object):
    __slots__ = ['traj','plain_text']

    def __init__(self, traj):
        self.traj = traj
        self.plain_text = tuple([len(self.traj)] + [step1['action']for step1 in traj])
    def __eq__(self, other):
        return self.plain_text == other.plain_text
    def __hash__(self):
        return hash(self.plain_text)

def max_tree_backprop():
    data = []
    # data += [[{'obs': (0,0), 'next_obs': (0,1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (0,1), 'next_obs': (1,0), 'action': 0, 'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
    #           {'obs': (1,0), 'next_obs': (1,1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (1,1), 'next_obs': (1,5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True,'done': True}] for i in range(9)]
    # data += [[{'obs': (0,0), 'next_obs': (0,1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (0, 1), 'next_obs': (1, 0), 'action': 0, 'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
    #           {'obs': (1, 0), 'next_obs': (1, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (1, 1), 'next_obs': (2, 0), 'action': 0,'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
    #           {'obs': (2, 0), 'next_obs': (2, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (2, 1), 'next_obs': (2, 5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True, 'done': True}]]
    # data += [[{'obs': (0,0), 'next_obs': (0,1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
    #           {'obs': (0, 1), 'next_obs': (0, 5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True, 'done': True}] for i in range(90)]
    data += [
        [{'obs': (0, 0), 'next_obs': (0, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 1), 'next_obs': (0, 5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True, 'done': True}]
        for i in range(90)]
    data += [
        [{'obs': (0, 0), 'next_obs': (0, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 1), 'next_obs': (0, 0), 'action': 0, 'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
         {'obs': (0, 0), 'next_obs': (0, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 1), 'next_obs': (0, 5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True, 'done': True}]
        for i in range(9)]
    data += [
        [{'obs': (0, 0), 'next_obs': (0, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 1), 'next_obs': (0, 0), 'action': 0, 'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
         {'obs': (0, 0), 'next_obs': (0, 1), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 1), 'next_obs': (0, 2), 'action': 0, 'reward': 0.7, 'value': 0., 'real_done': True, 'done': False},
         {'obs': (0, 2), 'next_obs': (0, 3), 'action': 0, 'reward': 0., 'value': 0., 'real_done': False, 'done': False},
         {'obs': (0, 3), 'next_obs': (0, 5), 'action': 0, 'reward': 0.4, 'value': 0., 'real_done': True, 'done': True}]]





    for traj in data:
        process_maxtree_g(traj)
    print()
    np.random.shuffle(data)
    from ding.worker import EpisodeSerialCollector
    from ding.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
    from ding.policy import DQNPolicy
    from ding.model import DQN
    from dizoo.classic_control.cartpole.envs import CartPoleEnv
    env = BaseEnvManager([lambda: CartPoleEnv({}) for _ in range(8)], BaseEnvManager.default_config())
    env.seed(0)
    model = DQN(obs_shape=4, action_shape=1)
    policy = DQNPolicy(DQNPolicy.default_config(), model=model).collect_mode
    collector = MaxTreeCollector(EpisodeSerialCollector.default_config(), env, policy)
    collector.process_max_tree(data)
    data2 = set(map(hash_traj,data))
    print(data2)


def reshape_test():
    a = torch.zeros((3,4,5,6))
    b = a.reshape(-1,30)
    print(b)
def sum_map():
    a = {'1': Qvalue(1), '2': Qvalue(2), '3': Qvalue(3)}
    b = sum(map(lambda x: x.v, a.values()))
    print()
def epsilon_greedy(prob,ep,mask):
    p = prob * (1-ep) + mask * ep/sum(mask)
    return p
def gumbel_sample(logits,mask=None):
    noise = np.random.gumbel(size=len(logits))
    gl = logits + noise
    if mask is not None:
        gl -= 1E8*(1-mask)
    sample = np.argmax(gl)
    # noise = torch.Tensor(logits.shape).uniform_() #tf.random_uniform(tf.shape(logits))
    # sample = torch.argmax(logits - torch.log(-torch.log(noise)), -1)
    return sample

def gumble_max():
    n_cats = 265
    mask = np.zeros((n_cats,))
    mask[[2,3,5]] = 1
    mask[40:67] = 1
    prob = torch.zeros((n_cats,),dtype=torch.float32)
    prob[2] = .2
    prob[3] = .3
    prob[5] = .5
    prob_ep = epsilon_greedy(prob,0.4,mask)
    prob_ep = prob_ep.numpy()
    # sample = np.random.choice(range(n_cats), p=prob_ep, size=100)
    for i in range(100):
        gumble_sample = np.array([gumbel_sample(np.log(prob_ep+1E-10),mask) for _ in range(100)])
        a2 = sum(gumble_sample == 2)
        a3 = sum(gumble_sample == 3)
        a5 = sum(gumble_sample == 5)
        z = a2+a3+a5
        print(f"{a2/z}+{a3/z}+{a5/z} = {z}")



M = 5
if __name__ == '__main__':
    max_tree_backprop()
    # compare_sort()