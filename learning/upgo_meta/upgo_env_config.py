from easydict import EasyDict

upgo_env_config = dict(
    exp_name='upgo_meta',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=3,
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=256),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
upgo_env_config = EasyDict(upgo_env_config)
# main_config = cartpole_dqn_config
# cartpole_dqn_create_config = dict(
#     env=dict(
#         type='cartpole',
#         import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
#     ),
#     env_manager=dict(type='base'),
#     policy=dict(type='dqn'),
# )
# cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
# create_config = cartpole_dqn_create_config
