from ding.policy import PPOPolicy

import sys


class h3_ppo_policy(PPOPolicy):
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
