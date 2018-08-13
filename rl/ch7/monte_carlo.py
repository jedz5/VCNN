import numpy as np
from contextlib import contextmanager
import time

from ch6.myTestEnv import myEnv, ModelFreeAgent, TableAgent, eval_game
import gym
from ch6.my_iter import PolicyIteration

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))

class MonteCarlo(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon
        self.i = 0

    def monte_carlo_eval(self, agent, env):
        state = env.reset()
        episode = []
        while True:
            ac = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(ac)
            episode.append((state, ac, reward))
            state = next_state
            if terminate:
                break

        value = []
        return_val = 0
        for item in reversed(episode):
            return_val = return_val + item[2]
            value.append((item[0], item[1], return_val))
        # every visit
        for item in reversed(value):
            agent.value_n[item[0]][item[1]] += 1
            agent.value_q[item[0]][item[1]] += (item[2] -  \
                agent.value_q[item[0]][item[1]]) /  \
                agent.value_n[item[0]][item[1]]
        # first visit
        

    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        new_policy = np.argmax(agent.value_q,axis=1)
        self.i += 1
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            diff = agent.pi - new_policy
            print("i = {}".format(self.i))
            print(diff)
            agent.pi = new_policy
            return True

    # monte carlo
    def monte_carlo_opt(self, agent, env):
        for i in range(100):
            for j in range(1):
                self.monte_carlo_eval(agent, env)
            self.policy_improve(agent)

def monte_carlo_demo():
    # np.random.seed(101)
    env = myEnv(0, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo()
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env,agent)))
    print(agent.pi)

    # np.random.seed(101)
    agent2 = TableAgent(env,0)
    pi_algo = PolicyIteration()
    with timer('Timer PolicyIter'):
        pi_algo.policy_iteration(agent2)
    print('return_pi={}'.format(eval_game(env,agent2)))
    print(agent2.pi)

def monte_carlo_demo2():
    # np.random.seed(101)
    env = myEnv(0, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(0.5)
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env,agent)))
    print(agent.pi)

if __name__ == '__main__':
    # monte_carlo_demo()
    monte_carlo_demo2()



