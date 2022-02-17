from ding.config import compile_config
from tensorboardX import SummaryWriter

import os

import torch

from ENV.H3_battle import Battle
from ding.worker import EpisodeSerialCollector, BaseLearner, NaiveReplayBuffer, \
    InteractionSerialEvaluator
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from ding.policy import PPOPolicy
from ding_model.H3EnvWrapper import DingH3Env,GymH3EnvWrapper
from ding_model.H3Q_model import H3Q_model
from ding_model.H3ppo_policy import h3_ppo_policy
from ding_model.h3_ppo_config import main_config


def train():
    cfg = main_config
    cfg = compile_config(
        cfg,
        AsyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        EpisodeSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collect_env = AsyncSubprocessEnvManager([lambda: DingH3Env(GymH3EnvWrapper(Battle(load_file='ENV/battles/0.json'))) for _ in range(1)], cfg.env.manager)
    eval_env = AsyncSubprocessEnvManager(
        [lambda: DingH3Env(GymH3EnvWrapper(Battle(load_file='ENV/battles/0.json'))) for _ in range(1)],
        cfg.env.manager)
    # env.seed(0)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    model = H3Q_model()
    policy = h3_ppo_policy(cfg.policy, model=model)
    collector = EpisodeSerialCollector(cfg.policy.collect.collector, collect_env, policy.collect_mode,tb_logger, exp_name=cfg.exp_name, instance_name='collector')
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator,
                                                eval_env, policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='evaluator')
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )

    max_iterations = 100
    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop_flag, reward, = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            if stop_flag:
                break
        # Sampling data from environments
        collected_episode = collector.collect()
        print(f"data len = {len(collected_episode)}")
        learner.train(collected_episode, collector.envstep)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()