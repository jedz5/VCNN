from collections import namedtuple
from ding.model import model_wrap
from ding.policy import PPOPolicy

import sys
from ding.policy.common_utils import default_preprocess_learn
from ding.torch_utils import to_device
from ding.utils import split_data_generator
from ding.utils.data import default_collate, default_decollate

import torch
from typing import Any, Dict
import torch.nn.functional as F

from ding_model.collect_utils import h3q_collate


class h3_SQL_policy(PPOPolicy):
    def _init_collect(self):
        self._unroll_len = self._cfg.collect.unroll_len
        self._action_space = self._cfg.action_space
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_multinomial_sample')
        self._collect_model.reset()
        self._recompute_adv = self._cfg.recompute_adv

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
            "real_done": timestep.info["real_done"]
        }
        return transition

    # def _get_train_sample(self, data: list):
    #     r"""
    #     reward = [2,3,4,5]
    #     prev_reward = reward[:-1] = [0,2,3,4]
    #     r = reward - prev_reward = [2,1,1,1]
    #     g = reward[-1] - prev_reward = [5,3,2,1]
    #     """
    #     last_done = data[-1]['real_done']
    #     last_done = int(last_done)
    #     if last_done < len(data) - 1:
    #         assert data[last_done]['real_done']
    #         data = data[:last_done + 1]
    #         print(f"end_reward={data[-1]['reward']}")
    #     end_reward = 0
    #     data[-1]['real_done'] = True
    #     prev_data = [None] + data[:-1]
    #     for step, prev_step in zip(data[::-1], prev_data[::-1]):
    #         # assert step['obs']['action_mask'][step['action']].item() == 1,f"action id = {step['action'].item()}"
    #         if step["real_done"]:
    #             end_reward += step['reward']
    #         if prev_step:
    #             step['g'] = end_reward - prev_step['reward'] * (1 - int(prev_step["real_done"]))
    #         else:
    #             step['g'] = end_reward
    #         step['adv'] = step['g'] - step['value']
    #     return data

    def _forward_collect(self, data: dict, eps: float = -1) -> dict:
        data_id = list(data.keys())
        data = h3q_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor', eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _forward_learn(self, data: Dict[str, Any]):
        data = h3q_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor')
                q_loss = F.l1_loss(output['logit'].gather(-1, batch['action'].unsqueeze(-1)).squeeze(-1), batch['q'], reduction="mean")
                # wv, we = self._value_weight, self._entropy_weight
                total_loss = q_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    # 'adv_max': adv.max().item(),
                    # 'adv_mean': adv.mean().item(),
                    'value_min': batch['q'].min().item(),
                    'value_max': batch['q'].max().item(),
                    # 'approx_kl': ppo_info.approx_kl,
                    # 'clipfrac': ppo_info.clipfrac,
                }
                return_infos.append(return_info)
        return return_infos