from collections import namedtuple, defaultdict
from typing import Union, Dict, Optional, Any, DefaultDict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from ding.model import model_wrap
from ding.policy.common_utils import default_preprocess_learn
from ding.rl_utils.a2c import a2c_loss, a2c_data
from ding.torch_utils import to_device
from ding.utils.data import default_collate
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer, InteractionSerialEvaluator
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import A2CPolicy
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn, get_train_sample
from torch.optim import SGD

from learning.upgo_meta.upgo_env_config import upgo_env_config
from ding.utils import SequenceType, squeeze
class upgo_env:
    def __init__(self,obs_shape,**param):
        self.obs = 0
        self.obs_shape = obs_shape
    def step(self,act):
        act = int(act)
        reward = 0.
        done = False
        if act == 0:
            self.obs = max(self.obs - 1, 0)
            if self.obs == self.obs_shape // 4:
                reward = .15
                done = True
            # elif self.obs == self.obs_shape // 2:
            #     reward = .35
            #     done = True
            # elif self.obs == self.obs_shape // 4 * 3:
            #     reward = .45
            #     done = True

        elif act == 1:
            # if self.obs == self.obs_shape // 4:
            #     reward = .15
            #     done = True
            # elif self.obs == self.obs_shape // 2:
            #     reward = .3
            #     done = True
            # elif self.obs == self.obs_shape // 4 * 3:
            #     reward = .4
            #     done = True
            if self.obs == self.obs_shape // 2:
                reward = .25
                done = True
        elif act == 2:
            self.obs = min(self.obs + 1, self.obs_shape)
        else:
            print(f"wrong act: {act}")
            exit(-1)
        if self.obs == self.obs_shape:
            reward = .99
            done = True
        return self.obs,reward,done,{}
    def reset(self):
        self.obs = 0
        return self.obs
    def seed(self,param):
        pass
    def close(self):
        pass

class step_horizen_env:
    def __init__(self,env,horizen,h_reward = -0.99):
        self.steps = 0
        self._env = env
        self.horizen = horizen
        self.h_reward = h_reward
    def step(self,act):
        obs, reward, done, info = self._env.step(act)
        self.steps += 1
        # if done:
        #     reward -= 0.01 * (self.steps - self._env.obs_shape)
        # elif self.steps > self.horizen:
        #     done = True
        #     reward = self.h_reward
        if self.steps > self.horizen and not done:
            done = True
            reward = self.h_reward
        # return self.steps*self._env.obs_shape + obs, reward, done, info
        return obs, reward, done, info
    def reset(self):
        self.steps = 0
        obs = self._env.reset()
        return 0
    def seed(self,param):
        pass
    def close(self):
        pass


class upgo_model(nn.Module):
    r"""
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        **param
    ) -> None:
        super(upgo_model, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.q_table = torch.zeros((15*obs_shape,action_shape),requires_grad=True)
        self.ac_table = torch.zeros((15*obs_shape,action_shape),requires_grad=True)
    def parameters(self, recurse: bool = True):
        return [self.q_table,self.ac_table]
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str,**param) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        # x = step_obs[..., 1]
        logits = self.ac_table.gather(0, x.unsqueeze(-1).expand((x.shape[0], self.action_shape)).long())
        return {'logit': logits}

    def compute_critic(self, x: torch.Tensor) -> Dict:
        # x = step_obs[..., 1]
        q_value = self.q_table.gather(0, x.unsqueeze(-1).expand((x.shape[0], self.action_shape)).long())
        value = q_value.max(dim=-1)[0]
        return {'value': value}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        # x = step_obs[...,1]
        logits = self.ac_table.gather(0, x.unsqueeze(-1).expand((x.shape[0], self.action_shape)).long())
        q_value = self.q_table.gather(0, x.unsqueeze(-1).expand((x.shape[0], self.action_shape)).long())
        value = q_value.max(dim=-1)[0]
        return {'logit': logits, 'value': value,"q_value":q_value}
def defaultdict_int():
    return defaultdict(int)
class upgo_policy(A2CPolicy):
    def __init__(self,*par,**par2):
        super(upgo_policy, self).__init__(*par,**par2)
        self.Q = defaultdict(defaultdict_int)
        self.V = defaultdict(int)
        self.sars_count = defaultdict(defaultdict_int)
    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        # Optimizer
        self._optimizer = SGD(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate
        )

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._adv_norm = self._cfg.learn.adv_norm
        self._grad_norm = self._cfg.learn.grad_norm

        # Main and target models
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
    def _get_train_sample(self, data: list):
        # with torch.no_grad():
        #     last_value = self._model.compute_critic(data[-1]['next_obs'].unsqueeze(-1))['value']
        # value = torch.stack([d['value'] for d in data]+ [last_value])
        # reward = torch.stack([d['reward'] for d in data])
        # upgo_ret = self.get_upgo_return_sparse_reward_simple(reward,value)
        # for i in range(len(data)):
        #     data[i]['upgo_ret'] = upgo_ret[i]
        #     data[i]['adv'] = upgo_ret[i] - value[i]
        # return get_train_sample(data, self._unroll_len)
        return data
    def get_upgo_return_sparse_reward_simple(self,rewards,value):
        result = torch.zeros_like(rewards)
        # Forced cutoff at the last one
        # result[-1, :] = (rewards[-1,:] + value[-1]) if (rewards[-1,:] + value[-1]) > value[-2] else value[-2]
        r'''
        G   G   G <- 求V
        S - S - S - T
            G+r G+r G+r <-求Q
        搞清楚(S,A,R,S',G)中的G到底对应哪个S 很重要
        '''
        result[-1, :] = rewards[-1, :]
        for t in reversed(range(rewards.size()[0] - 1)):
            result[t, :] = result[t+1,:] if result[t+1,:] > value[t+1] else value[t+1]
        return result
    def get_upgo_return_sparse_reward_tree_max(self,data:list):
        self.clear_table()
        first_done = False
        for i,t in enumerate(reversed(data)):
            s = t['obs'].int().item() #tuple(t['obs'].int().numpy())
            a = t['action'].int().item() #tuple(t['action'].int().numpy())
            s_ = t['next_obs'].int().item() #tuple(t['next_obs'].int().numpy())
            if t['done']:
                first_done = True
                self.sars_count[s, a][(0,0,0,s,a)] += 1
                G = t['reward'].item()
                if G > .5:
                    print()
                # T = (0,0,0,s,a)
                # self.V[T] += (G - self.V[T]) / self.sars_count[s, a][T]
            else:
                if not first_done:
                    continue
                self.sars_count[s, a][s_] += 1
                G = max(self.Q[s_].values())
                if self.V[s_] > G:
                    print()
                G = max([self.V[s_],G,t['value']])
                self.V[s_] = .9 * G
            self.Q[s][a] = .9 * G
            # self.Q[s][a] = sum([self.sars_count[(s, a)][s_] * self.V[s_] for s_ in self.sars_count[(s, a)].keys()]) / sum(
            #     self.sars_count[(s, a)].values())
        return self.Q_to_sars()
    def Q_to_sars(self):
        sars = []
        for s,aq in self.Q.items():
            for a,q in aq.items():
                sars.append({'obs':torch.tensor(s,dtype=torch.float32),'action':torch.tensor(a,dtype=torch.long),'return_':torch.tensor(q,dtype=torch.float32)}) #'next_obs':torch.tensor(s_)
        return sars
    def clear_table(self):
        self.V.clear()
        self.Q.clear()
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        # forward
        output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')

        # adv = data['adv']
        # if self._adv_norm:
        #     # norm adv in total train_batch
        #     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # with torch.no_grad():
        #     return_ = data['value'] + adv
        return_ = data['return_']
        adv = data['return_'] - output['value'].detach()
        data = a2c_data(output['logit'], data['action'], output['q_value'], adv, return_, None)

        # Calculate A2C loss
        a2c_loss = upgo_a2c_error(data)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = a2c_loss.policy_loss + a2c_loss.value_loss - a2c_loss.entropy_loss

        # ====================
        # A2C-learning update
        # ====================

        self._optimizer.zero_grad()
        total_loss.backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     list(self._learn_model.parameters()),
        #     max_norm=self._grad_norm,
        # )
        self._optimizer.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            # 'grad_norm': grad_norm,
        }
def upgo_a2c_error(data: namedtuple) -> namedtuple:
    """
    Overview:
        Implementation of A2C(Advantage Actor-Critic) (arXiv:1602.01783)
    Arguments:
        - data (:obj:`namedtuple`): a2c input data with fieids shown in ``a2c_data``
    Returns:
        - a2c_loss (:obj:`namedtuple`): the a2c loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - value (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    """
    logit, action, q_value, adv, return_,weight = data
    # if weight is None:
    #     weight = torch.ones_like(q_value)
    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)
    value = q_value.gather(1,action.unsqueeze(-1)).squeeze(-1)
    entropy_loss = (dist.entropy()).sum()
    policy_loss = -(logp * adv).sum()
    value_loss = (F.mse_loss(return_, value, reduction='sum'))#.sum()
    return a2c_loss(policy_loss, value_loss, entropy_loss)


# Get DI-engine form env class

def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        A2CPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=False
    )
    def wrapped_upgo_env():
        return DingEnvWrapper(step_horizen_env(upgo_env(cfg.policy.model.obs_shape), 12, -1.))

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_upgo_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_upgo_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    # Set random seed for all package and instance
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = upgo_model(**cfg.policy.model)
    policy = upgo_policy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # Training & Evaluation loop
    while True:
        # Evaluating at the beginning and with specific frequency
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            print("q_table:")
            print(model.q_table.detach().numpy())
            if stop:
                break
        # Update other modules
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data = collector.collect(train_iter=learner.train_iter)
        # replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        #model.q_table.detach().numpy()
        # Training

        tree_data = policy.get_upgo_return_sparse_reward_tree_max(new_data)
        analy_tree(tree_data)
        print()
        for i in range(cfg.policy.learn.update_per_collect):
            # train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            # if train_data is None:
            #     break
            learner.train(tree_data, collector.envstep)
def analy_tree(tree_data):
    for t in tree_data:
        t['1_obs'] = (t['obs'].int().item()//6,t['obs'].int().item()%6)

if __name__ == "__main__":
    main(upgo_env_config)
