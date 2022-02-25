from collections import defaultdict
from ding.utils.data import default_collate
from ding.worker import EpisodeSerialCollector
import numpy as np
from functools import cmp_to_key
# from typing import Optional, Any, List, Tuple
# import torch
# from tianshou.data.batch import Batch
from ding_model.collect_utils import process_standard_g, indexToAction_simple


class Value:
    def __init__(self,v=0.):
        self.v = v
    def __repr__(self):
        return f"{self.v}"
def defaultdict_value():
    return defaultdict(Value)
def get_tuple(attri_stack_orig):
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
class MaxTreeCollector(EpisodeSerialCollector):
    def __init__(self,*args,**kwargs):
        cfg = args[0]
        assert cfg.get_train_sample == False
        super(MaxTreeCollector, self).__init__(*args,**kwargs)
        self.best_buffer = []
        self.Q = defaultdict(defaultdict_value) #Q[s] = {a,q}
        self.V = defaultdict(lambda : Value(-100.))
        self.sars_count = defaultdict(defaultdict_value)
        self.S = {}
        self.sa_count = defaultdict(lambda : Value(0))
    def process_max_tree(self,data:list):
        for i,traj in enumerate(data):
            for j, t in enumerate(reversed(traj)):
                s = tuple(map(tuple, t['obs']['attri_stack']))
                a = t['action']  # {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                s_ = tuple(map(tuple, t['obs_next']['attri_stack']))
                if t['done']:
                    first_done = True
                    T = (0, 0, 0, a, s)
                    n = self.sars_count[a, s][T]
                    n.v += 1
                    r = t['rew']
                    r'''step1 Vt = E(r+γ* Gt)= E(r) 因为Gt=0'''
                    self.V[T].v += (r - self.V[T].v) / n.v
                else:
                    if not first_done:
                        continue
                    self.sars_count[a, s][s_].v += 1
                    vs_ = max(self.Q[s_].values())  # TODO value compare
                    r'''step1 Vs_ = max(Vs_,max(Qs_a))'''
                    self.V[s_].v = max([self.V[s_].v, vs_])
                self.sa_count[a, s].v += 1
                r'''step3                                            step2  V[s_] == Gs_, 其他s_下的r=0   '''
                qvalue = sum([self.sars_count[a, s][s_].v * (self.V[s_].v - self.discount_factor) for s_ in
                              self.sars_count[a, s].keys()]) / \
                         sum(self.sars_count[a, s].values())
                qsa = self.Q[s][a]
                qsa.v = qvalue
                t.q = qsa
    def cmp_single_state(self,sars1,sars2):
        q1 = self.get_end_Q(sars1)
        q2 = self.get_end_Q(sars2)
        error1 = q1 - q2
        if abs(error1) < 1E-6:
            error1 = 0
        return error1
    def cmp_reward_func(self,episode1,episode2):
        assert episode1[-1]['done']
        assert episode2[-1]['done']
        # length = min(len(episode1.rew),len(episode2.rew))
        # for i in range(length):
        #     err = self.cmp_single_state(episode1[i],episode2[i])
        #     if abs(err) > 1E-6:
        #         return err
        # return 0
        err = episode1[0]['g'] - episode2[0]['g']
        if abs(err) > 1E-6:
            return err
        return 0
    def re_compute_ac(self,buffer):
        if len(buffer) == 0:
            return
        for traj in self.best_buffer:
            traj2 = default_collate(traj)
            policy_output = self._policy.forward({0:traj2['obs']})[0]
            for step,step_value,step_logit in zip(traj,policy_output['value'],policy_output['logit']):
                step['value'] = step_value
                step['logit'] = step_logit
    def collect(self,*args,**kwargs):
        trajs = super(MaxTreeCollector, self).collect(*args,**kwargs)
        self.re_compute_ac(self.best_buffer)
        trajs.extend(self.best_buffer)
        data = []
        for t in trajs:
            t1 = process_standard_g(t,self._logger)
            data.append(t1)
        data.sort(key=cmp_to_key(self.cmp_reward_func),reverse=True)
        self.best_buffer = data[:50]
        for t in self.best_buffer[:5]:
            act_str = "O"
            for step in t:
                act_str +=f"->{indexToAction_simple(step['action'])} "
                if step["real_done"]:
                    act_str +="T "
            self._logger.info(act_str)
            self._logger.info(f"best buffer reward = {t[0]['g']}")
        trajs.clear()
        for t in data:
            trajs.extend(t)
        return trajs




        