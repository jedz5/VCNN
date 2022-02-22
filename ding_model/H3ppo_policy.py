from ding.model import model_wrap
from ding.policy import PPOPolicy

import sys
from ding.torch_utils import to_device
from ding.utils.data import default_collate, default_decollate

import torch


class h3_ppo_policy(PPOPolicy):
    def _init_collect(self):
        self._unroll_len = self._cfg.collect.unroll_len
        self._action_space = self._cfg.action_space
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
    def _get_train_sample(self, data: list):
        end_reward = data[-1]['reward'].item()
        last_rew = 0
        if end_reward < -1:
            last_done = int(-end_reward)
            # assert data[last_done]['reward'].item() > 0.3
            data = data[:last_done+1]
            print(f"end_reward={data[-1]['reward']}")
        end_reward = data[-1]['reward']
        for step in data:
            assert step['obs']['action_mask'][step['action']].item() == 1,f"action id = {step['action'].item()}"
            step['g'] = end_reward - last_rew
            step['adv'] = end_reward - last_rew - step['value']
            last_rew = step['reward']
        return data

    def _forward_collect(self, data: dict,eps:float = -1) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic',eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}