from ENV.H3_battle import Battle
from ding.worker import EpisodeSerialCollector
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from ding.policy import PPOPolicy
from ding_model.H3EnvWrapper import DingH3Env,GymH3EnvWrapper
from ding_model.H3Q_model import H3Q_model




def train():
    env = BaseEnvManager([lambda: DingH3Env(GymH3EnvWrapper(Battle(load_file='ENV/battles/0.json'))) for _ in range(1)], BaseEnvManager.default_config())
    env.seed(0)
    model = H3Q_model()
    policy = PPOPolicy(PPOPolicy.default_config(), model=model).collect_mode
    collector = EpisodeSerialCollector(EpisodeSerialCollector.default_config(), env, policy)

    collected_episode = collector.collect(
        n_episode=2, train_iter=collector._collect_print_freq
    )
    assert len(collected_episode) == 2
    assert all([e[-1]['done'] for e in collected_episode])
    assert all([len(c) == 0 for c in collector._traj_buffer.values()])

if __name__ == '__main__':
    train()