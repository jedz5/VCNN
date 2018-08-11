# value iteration
import numpy as np
from contextlib import contextmanager
import time
from ch6.myTest import myEnv, TableAgent, eval_game

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))

def value_iteration_demo():
    np.random.seed(0)
    env = myEnv(0, [3,6])
    agent = TableAgent(env,0)
    vi_algo = ValueIteration()
    vi_algo.value_iteration(agent)
    print('return_pi={}'.format(eval_game(env,agent)))
    print(agent.pi)


class ValueIteration(object):
    def value_iteration(self, agent, max_iter = -1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = np.zeros_like(agent.value_pi)
            R = agent.r + agent.gamma * agent.value_pi
            # for each state
            value_sas = np.zeros([agent.s_len,2])
            for i in reversed(range(1, agent.s_len)):
                for j in range(0, agent.a_len): # for each act
                    value_sa = np.dot(agent.p[j, i, :], R)
                    value_sas[i][j] = (value_sa)
            new_value_pi = np.max(value_sas,axis=1)
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break
        print('Iter {} rounds converge'.format(iteration))
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i,j] = np.dot(agent.p[j,i,:], agent.r + agent.gamma * agent.value_pi)
        max_act = np.argmax(agent.value_q,axis=1)
        agent.pi = max_act

if __name__ == '__main__':
    value_iteration_demo()