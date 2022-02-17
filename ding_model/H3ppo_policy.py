from ding.policy import PPOPolicy

import sys


class h3_ppo_policy(PPOPolicy):
    def _get_train_sample(self, data: list):
        end_reward = data[-1]['reward']
        last_rew = 0
        if end_reward < -1:
            for step in data:
                print(step['reward'])
            sys.exit(0)
        for step in data:
            step['g'] = end_reward - last_rew
            step['adv'] = end_reward - last_rew - step['value']
            last_rew = step['g']
        return data