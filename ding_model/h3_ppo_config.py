from easydict import EasyDict

import platform

Linux = "Linux" == platform.system()
env_n = 32 if Linux else 4
gobigger_config = dict(
    exp_name='h3_ppo',
    env=dict(
        collector_env_num=env_n,
        evaluator_env_num=8 if Linux else 8,
        evaluator_env_num2=8 if Linux else 8,
        n_evaluator_episode=32,
        stop_value=3,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        recompute_adv=False,
        # action_space='continuous',
        # model=dict(
        #     obs_shape=11,
        #     action_shape=3,
        #     action_space='continuous',
        # ),
        learn=dict(
            epoch_per_collect=4,
            batch_size=2048 if Linux else 64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.5,
            clip_ratio=0.2,
            adv_norm=False,
            value_norm=False,
        ),
        collect=dict(
            n_episode=1024 if Linux else 32,
            unroll_len=1,#collector=dict(get_train_sample=True, )
            env_n = env_n
        ),
        # collect=dict(
        #     n_sample=2048,
        #     unroll_len=1,
        #     discount_factor=0.99,
        #     gae_lambda=0.97,
        # ),
        eval=dict(evaluator=dict(eval_freq=300,n_episode=32,stop_value=3, ),evaluator2=dict(eval_freq=300,n_episode=1,stop_value=3, )),
    )
)
main_config = EasyDict(gobigger_config)
# gobigger_create_config = dict(
#     env=dict(
#         type='gobigger',
#         import_names=['dizoo.gobigger.envs.gobigger_env'],
#     ),
#     env_manager=dict(type='subprocess'),
#     policy=dict(type='dqn'),
# )
# create_config = EasyDict(gobigger_create_config)