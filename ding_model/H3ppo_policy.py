from ding.policy import PPOPolicy


class h3_ppo_policy(PPOPolicy):
    def _get_train_sample(self, data: list):
        # np.add.accumulate(batch_rew[batch_done > 0][::-1])
        return data