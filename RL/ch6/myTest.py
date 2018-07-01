import numpy as np
N = 7
import gym
from gym.spaces import Discrete

class myEnv(gym.Env):
    SIZE = N

    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(dices))
        tmpladders = {}
        for k, v in self.ladders.items():
            tmpladders[v] = k
            tmpladders[k] = v
        self.ladders = tmpladders
        self.pos = 1
        self.R = N+3
        print('ladders info:')
        print(self.ladders)
        print('dice ranges:')
        print(self.dices)

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == N:
            return N, N, 1, {}
        elif self.pos > N:
            self.pos = N * 2 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def reward(self, s):
        if s == N:
            return N
        else:
            return -1

    def render(self):
        pass


class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.r = [env.reward(s) for s in range(0, self.s_len)]
        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, N):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step > N, step <= N],
                                    [lambda x: N * 2 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob
        self.p[:, N, N] = 1
        self.value_pi = np.zeros((self.s_len))
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]
if __name__ == '__main__':
    env = myEnv(0, [3,4])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(1)
        print(reward, state)
        if terminate == 1:
            break