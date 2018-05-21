import numpy as np
from ch6.snake import SnakeEnv, TableAgent, eval_game

policy_ref = [1] * 97 + [0] * 3
policy_0 = [0] * 100
policy_1 = [1] * 100

def test_easy():
    np.random.seed(0)
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0
    env = SnakeEnv(0, [3, 6])
    for i in range(10000):
        sum_opt += eval_game(env, policy_ref)
        sum_0 += eval_game(env, policy_0)
        sum_1 += eval_game(env, policy_1)
    print('opt avg={}'.format(sum_opt / 10000.0))
    print('0 avg={}'.format(sum_0 / 10000.0))
    print('1 avg={}'.format(sum_1 / 10000.0))

class PolicyIteration(object):

    def policy_evaluation(self, agent, max_iter = -1):
        iteration = 0
        # iterative eval
        while True:
            # one iteration
            iteration += 1
            new_value_pi = agent.value_pi.copy()
            for i in range(1, agent.s_len): # for each state
                value_sas = []
                ac = agent.pi[i]
                # for j in range(0, agent.act_num): # for each act
                # print ac
                transition = agent.p[ac, i, :]
                value_sa = np.dot(transition, agent.r + agent.gamma * agent.value_pi)
                    # value_sas.append(value_sa)
                new_value_pi[i] = value_sa# value_sas[agent.policy[i]]
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            # print 'diff={}'.format(diff)
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break

    def policy_improvement(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i,j] = np.dot(agent.p[j,i,:], agent.r + agent.gamma * agent.value_pi)
                # update policy
            max_act = np.argmax(agent.value_q[i,:])
            new_policy[i] = max_act
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def policy_iteration(self, agent):
        iteration = 0
        while True:
            iteration += 1
            self.policy_evaluation(agent)
            ret = self.policy_improvement(agent)
            if not ret:
                break
        print('Iter {} rounds converge'.format(iteration))


def policy_iteration_demo1():
    env = SnakeEnv(0, [3,6])
    agent = TableAgent(env)
    agent.pi = np.array([1 for s in range(0, agent.s_len)])
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)

def policy_iteration_demo2():
    env = SnakeEnv(0, [3,6])
    agent = TableAgent(env)
    agent.pi[:]=0
    print('return3={}'.format(eval_game(env,agent)))
    agent.pi[:]=1
    print('return6={}'.format(eval_game(env,agent)))
    agent.pi[97:100]=0
    print('return_ensemble={}'.format(eval_game(env,agent)))
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env,agent)))
    print(agent.pi)
def xxx():
    # a = np.matrix('0.25 0.25 0.25 0.25 0 0 ; 0 0.25 0.25 0.25 0.25 0 ; 0 0 0.25 0.25 0.25 0.25 ; 0 0 0 0.25 0.5 0.25 ; 0 0 0 0.25 0.5 0.25 ; 0 0 0 0 0 1')
    # a = a.T
    # print(a)
    # a = a ** 20
    # np.set_printoptions(precision=2)
    # print(a)
    # b = np.matrix('1 0 0 0 0 0').T
    # print(np.floor(a*b))
    cs = cp = 0.0025
    def close(mon,t,new_v):
        m = mon[t]
        m[1] = m[0]
        m[1,1] = new_v
        omega(m)
        if(t < len(mon) - 1):
            mon[t+1][0] = m[1]
    def omega(m,y = True):
        if y:
            m[1,3] = m[1,1] / (m[0,1]+1.0e-10) #y
            tmp = (m[0,2] * m[1,3])
            m[1,2] = tmp / (sum(tmp)+1.0e-10)
        else:
            tmp = m[0,0] * m[0,1]
            m[0, 2] = tmp / (sum(tmp) + 1.0e-10)
    def sell(mon,howmany,which):
        if howmany == 0:
            return
        m = mon[0]
        sold = m[0,which] * howmany  #份数
        m[0, which] -= sold
        m[0,0] += sold * m[1,which] * (1 - cs)
        omega(mon, False)
    def buy(mon,howmany,which):
        if howmany == 0:
            return
        m = mon[0]
        bought = m[0,0] * howmany #现金
        m[0,0] -= bought
        m[0,which] += bought * (1-cp) / m[1,which]
        omega(mon,False)
    Day = 3
    money = np.zeros((Day,2,4,5)) #样本,开盘/收盘,份/元/omega/比,现金/资产1/资产2/资产3/资产4
    money[0,0,0] = [0,1000,1000,1000,1000] #份
    money[0,0,1] = [1,1,1,1,1] #元
    money[0,0,2] = [0,0.25,0.25,0.25,0.25] #omega
    money[0,0,3] = [1,1,1,1,1] #比

    value = [[1,2,0.5,1,1],[1, 1, 1, 2, 1],[1,1,3,1,1]]
    to_sell = [(0,0),(0.3,2), (1.0,1)]
    to_buy = [(0,0),(1.0,3), (0.5,3)]
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    pp = 1
    for x in range(Day):
        print('day {} open:'.format(x))
        print(money[x,0])
        if x > 0:
            print('---------after sell {}'.format(to_sell[x]))
            sell(money[x], to_sell[x][0], to_sell[x][1])
            if x == 1:
                sell(money[x], 0.3, 4)
            print(money[x, 0])
            print('----after buy {}'.format(to_buy[x]))
            buy(money[x], to_buy[x][0], to_buy[x][1])
            print(money[x, 0])
            v0 = sum(money[x, 0, 0] * money[x, 0, 1])
            v1 = sum(money[x - 1, 0, 0] * money[x - 1, 0, 1])
            # pt = sum(money[x - 1, 0, 2] * money[x, 0, 3])
            pt = v0 / v1
            print('Pt: %2f = %2f / %2f' % (pt, v0, v1))
            miu = pt/pp
            print('μ:{} = {} / {}'.format(miu,pt,pp))
            sold = (1 - cs) *pp *np.sum(np.maximum(money[x - 1,1,2,1:] - miu*money[x,0,2,1:],0))
            print('sold = {}'.format(sold))
        print('day {} close:'.format(x))
        close(money,x,value[x])
        print(money[x, 1])
        v0 = sum(money[x,1,0]*money[x,1,1])
        v1 = sum(money[x,0,0]*money[x,0,1])
        pp = sum(money[x,0,2] * money[x,1,3])
        print('P\': %2f = %2f / %2f'%(pp,v0,v1))
if __name__ == '__main__':
    xxx()


