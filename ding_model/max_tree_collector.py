from collections import defaultdict
from ding.utils.data import default_collate
from ding.worker import EpisodeSerialCollector
import numpy as np
from functools import cmp_to_key
# from typing import Optional, Any, List, Tuple
# import torch
from ding_model.collect_utils import process_standard_g, indexToAction_simple, Value, defaultdict_value, h3q_collate


class MaxTreeCollector(EpisodeSerialCollector):
    def __init__(self,*args,**kwargs):
        cfg = args[0]
        assert cfg.get_train_sample == False
        super(MaxTreeCollector, self).__init__(*args,**kwargs)
        self.best_buffer = []
        self.Q = defaultdict(defaultdict_value) #Q[s] = {a,q}
        self.V = defaultdict(lambda : Value(-100.))
        self.sars_count = defaultdict(defaultdict_value)
        # self.S = {}
        self.sa_count = defaultdict(lambda : Value(0))
    def process_max_tree(self,data:list):
        for i,traj in enumerate(data):
            for j, t in enumerate(reversed(traj)):
                s = tuple(map(tuple, t['obs']['attri_stack'].numpy()))
                a = t['action'].item()  # {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                s_ = tuple(map(tuple, t['next_obs']['attri_stack'].numpy()))
                if t['done']:
                    first_done = True
                    T = (0, 0, 0, a, s)
                    n = self.sars_count[a, s][T]
                    n.v += 1
                    r = t['reward'].item()
                    r'''step1 Vt = E(r+γ* Gt)= E(r) 因为Gt=0'''
                    self.V[T].v += (r - self.V[T].v) / n.v
                else:
                    if not first_done:
                        continue
                    n = self.sars_count[a, s][s_]
                    n.v += 1
                    vs_ = max(map(lambda x:x.v,self.Q[s_].values()))
                    r'''step1 Vs_ = max(Vs_,max(Qs_a))'''
                    self.V[s_].v = max([self.V[s_].v, vs_])
                    r'''
                   pre_step2  预备工作
                   Gs = rss' + Gs_ - self.discount_factor
                   为保证rss'不随策略pi改变self,一种最简单实现方式为中间reward统统=0
                   只有real_done=True时，r != 0
                   '''
                    r = t['reward'].item()
                    self.V[s_].v += r
                nsa = self.sa_count[a, s]
                nsa.v += 1
                r'''step3                                            step2  V[s_] == Gs_, 其他s_下的r=0   '''
                qvalue = sum([self.sars_count[a, s][ss_].v * self.V[ss_].v for ss_ in
                              self.sars_count[a, s].keys()]) / \
                         sum(map(lambda x:x.v,self.sars_count[a, s].values()))
                qsa = self.Q[s][a]
                qsa.v = qvalue
                r'''对应real_done=True'''
                if not t['done']:
                    self.V[s_].v -= r
                t['q'] = qsa
                t['nsa'] = nsa
                t['nsas'] = n
        return data
    def cmp_single_state(self,step1,step2):
        q1 = step1['q'].v
        q2 = step2['q'].v
        error1 = q1 - q2
        if abs(error1) < 1E-6:
            error1 = 0
        return error1
    def cmp_reward_func(self,episode1,episode2):
        assert episode1[-1]['done']
        assert episode2[-1]['done']
        length = min(len(episode1),len(episode2))
        for i in range(length):
            err = self.cmp_single_state(episode1[i],episode2[i])
            if abs(err) > 1E-3:
                return err
        return 0
        # err = episode1[0]['g'] - episode2[0]['g']
        # if abs(err) > 1E-6:
        #     return err
        # return 0
    def re_compute_ac(self,buffer):
        if len(buffer) == 0:
            return
        for traj in self.best_buffer:
            traj2 = h3q_collate(traj)
            policy_output = self._policy.forward({0:traj2['obs']})[0]
            for step,step_logit in zip(traj,policy_output['logit']):
                # step['value'] = step_value
                step['logit'] = step_logit
    def collect(self,*args,**kwargs):
        trajs = super(MaxTreeCollector, self).collect(*args,**kwargs)
        # self.re_compute_ac(self.best_buffer)
        trajs.extend(self.best_buffer)
        # data = []
        # for t in trajs:
            # t1 = process_standard_g(t,self._logger)
            # data.append(t1)
        self.process_max_tree(trajs)
        trajs.sort(key=cmp_to_key(self.cmp_reward_func),reverse=True)
        self.best_buffer = trajs[:50]
        self._logger.info("-----------------------------------------------------------")
        for t in self.best_buffer[:5]:
            act_str = "O"
            for step in t:
                stack_orig = tuple(step['obs']['attri_stack_orig'][0,2:5].numpy())
                stack = tuple(step['obs']['attri_stack'][0, 3:7].numpy())
                act_str +=f"->[{stack_orig[0]},{stack_orig[1]},{stack_orig[2]},{stack[0]},{stack[2]},{stack[3]}]{indexToAction_simple(step['action'])} {step['nsas'].v}/{step['nsa'].v} "
                if step["real_done"]:
                    act_str +="T "
            self._logger.info(act_str)
            self._logger.info(f"head/tail/reward/ = {t[0]['q']}/{t[-1]['q']}/{t[-1]['reward'].item()}")
        data = []
        for t in trajs:
            data.extend(t)
        return data




        