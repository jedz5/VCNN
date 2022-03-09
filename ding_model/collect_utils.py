import numpy as np
import re
from collections import defaultdict
from collections.abc import Sequence, Mapping
from typing import List, Dict, Union, Any
from torch._six import string_classes
import collections.abc as container_abcs
from ding.compatibility import torch_gt_131

import torch
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)
int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')
class Value:
    def __init__(self,v=0):
        self.v = v
    def __repr__(self):
        return f"{self.v}"
def defaultdict_value():
    return defaultdict(Value)

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
                logger.info(f"end_reward={data[-1]['reward']}")
                end_reward = max(end_reward,0) #最后一场的负值不能传递到前面场次
                logger.info(f"current_reward={end_reward}")
            end_reward += step['reward']
            win = True
        if prev_step:
            step['g'] = end_reward - prev_step['reward'] * (1 - int(prev_step["real_done"]))
        else:
            step['g'] = end_reward
        step['adv'] = step['g'] - step['value']
    return data
def process_maxtree_g(data:list,compare_equal_func):
    last_start = 0
    reward_cum = 0
    for i,step in enumerate(data):
        if step['real_done']:
            if i+1 < len(data):
                if compare_equal_func(data[last_start],data[i+1]):
                    reward_cum += step['reward']
                    step['reward'] -= step['reward']
                else:
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
def get_normalized_obs(attri_stack_orig):
    # attri_stack[i] = np.array(
    #     [st.side, st.slotId, st.id, st.amount, st.amount_base, st.first_HP_Left, st.health, st.luck, st.attack,
    #      st.defense, st.max_damage, st.min_damage,
    #      st.speed, st.morale, st.shots, st.y, st.x, int(st.had_moved), int(st.had_waited), int(st.had_retaliated),
    #      int(st.had_defended)])
    # mask = attri_stack_orig[...,9] > 10
    # if mask.any():
    #     print()
    attri_stack = np.copy(attri_stack_orig)
    r'''部队基础总生命值'''
    health_baseline = attri_stack[..., 4] * attri_stack[..., 6]
    health_baseline = health_baseline.astype('int')
    r'''部队基础总生命值中最大值'''
    health_baseline_max = np.max(health_baseline,axis=-1)
    r'''部队当前总生命值'''
    health_current = np.clip(attri_stack[..., 3] - 1, 0, np.inf) * attri_stack[..., 6] + attri_stack[..., 5]
    health_current = health_current.astype('int')
    r'''部队基础总生命值 / 部队基础总生命值中最大值'''
    health_ratio_bymax = health_baseline * 10 // health_baseline_max[...,None]
    r'''部队当前总生命值 / 部队基础总生命值'''
    health_current_ratio = (health_current * 10 // (health_baseline + 1E-9)).astype('int') + (
            health_current > 0).astype('int')
    r'''部队当前数量 / 部队基础数量'''
    amount_ratio = (attri_stack[..., 3] * 10 // (attri_stack[..., 4] + 1E-9)).astype('int') + (
                attri_stack[..., 3] > 0).astype('int')
    r'''远程弹药数量 / 基数16'''
    shoots_ratio = (attri_stack[..., 14] * 4 // 16.0001).astype('int') + (attri_stack[..., 14] > 0).astype('int')
    attri_stack[..., 3] = amount_ratio
    attri_stack[..., 4] = 0
    attri_stack[..., 5] = health_current_ratio
    attri_stack[..., 6] = health_ratio_bymax
    attri_stack[..., 14] = shoots_ratio


    # attri_stack = tuple(map(tuple, attri_stack))
    return attri_stack
bFieldWidth = 17
bFieldSize = 15 * 11
def indexToAction_simple(step):
    move = int(step['action'])
    if (move < 0):
        print('wrong move {}'.format(move))
        exit(-1)
    if (move == 0):
        return "w"
    elif (move == 1):
        return "d"
    elif ((move - 2) >= 0 and (move - 2) < bFieldSize):
        y = (move - 2) // (bFieldWidth - 2)
        x = (move - 2) % (bFieldWidth - 2) + 1
        return f"m({y},{x})"
    elif ((move - 2 - bFieldSize) >= 0 and (move - 2 - bFieldSize) < 14):
        enemy_id = move - 2 - bFieldSize
        stack_orig = tuple(step['obs']['attri_stack_orig'][enemy_id, 2:5].numpy())
        # stack = tuple(step['obs']['attri_stack'][enemy_id, 3:7].numpy())
        return f"sh({enemy_id}[{stack_orig[1]}/{stack_orig[2]}])"
    elif ((move - 2 - bFieldSize - 14) >= 0):
        direction = (move - 2 - bFieldSize - 14) % 6
        enemy_id = (move - 2 - bFieldSize - 14) // 6
        stack_orig = tuple(step['obs']['attri_stack_orig'][enemy_id, 2:5].numpy())
        return f"att({enemy_id}[{stack_orig[1]}/{stack_orig[2]}],d{direction})"
    else:
        print('wrong move {}'.format(move))
        exit(-1)

def h3q_collate(batch: Sequence,
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
            return h3q_collate([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, Value):
        return torch.tensor([x.v for x in batch], dtype=torch.float32)
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
                ret[key] = h3q_collate([d[key] for d in batch], cat_1dim=cat_1dim)
        return ret
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(h3q_collate(samples, cat_1dim=cat_1dim) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [h3q_collate(samples, cat_1dim=cat_1dim) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
