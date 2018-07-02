import numpy as np
import gym
from gym.spaces import Discrete

N = 10
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
            # print 'ladders info:'
            # print self.ladders
            # print 'dice ranges:'
            # print self.dices
        self.ladders = tmpladders
        self.pos = 1
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


class ModelFreeAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.value_n = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return self.pi[state]


def eval_game(env, policy):
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy, TableAgent) or isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise Exception('Illegal policy')
        state, reward, terminate, _ = env.step(act)
        # print state
        return_val += reward
        if terminate:
            break
    return return_val


if __name__ == '__main__':
    env = myEnv(10, [3, 6])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(0)
        print(reward, state)
        if terminate == 1:
            break





