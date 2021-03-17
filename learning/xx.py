import gym


env = gym.make('MontezumaRevengeDeterministic-v4')
ob = env.step(1)
env.