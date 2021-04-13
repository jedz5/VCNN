import numpy as np
import gym
from gym.spaces import Discrete
from torch import nn
from learning.my_tool import np_hash
from learning.gumoko_env import Board
import copy
np.set_printoptions(precision=3)
N = 70
class myEnv(gym.Env):
    SIZE = N

    def __init__(self, ladder_num, dices):
        self.dices = dices
        # self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(dices))
        # tmpladders = {}
        # for k, v in self.ladders.items():
        #     tmpladders[v] = k
        #     tmpladders[k] = v
        #     # print 'ladders info:'
        #     # print self.ladders
        #     # print 'dice ranges:'
        #     # print self.dices
        # self.ladders = tmpladders
        self.ladders = ladder_num
        self.pos = 1
        # print('ladders info:')
        # print(self.ladders)
        # print('dice ranges:')
        # print(self.dices)

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.choice(self.dices[a])
        self.pos += step
        if self.pos == N:
            return N, 100, 1, {}
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
# policy = [[1,2],[3,4],[5,6],[7,7,1]]
# lad = {4:20,20:4,16:35,35:16,50:80,80:50,98:3}
lad = {}
# policy = [[1,2,3],[1,2,3,4,5,6]]
# Q = np.zeros((N+1, policy_len))
# Pi = np.zeros((N+1, policy_len)) + 1/policy_len
height = 6
policy_len = height*height
gamma = 1.0
Lr = 1.
class state_node:
    def __init__(self,s,a_node,Qsa,Ps):
        self.s = s
        self.prev_a = a_node
        self.acts = {} #act_node(i,self) for i in range(policy_len)
        self.q = Qsa.copy()
        self.pi = Ps.copy()
        self.v = -100
        self.r = 0
        self.n = 0
    def select(self):
        a = np.random.choice(range(policy_len),p=self.pi)
        if a not in self.acts:
            act_n = act_node(a,self)
            self.acts[a] = act_n
        return a
    def back_up_v(self):
        self.n += 1
        self.v = np.max(self.q)
        # print(f"here {self.s} is {self.v}")
        pi2 = update_Pi(self.pi,self.q,Lr)
        self.pi = pi2
        if self.prev_a:
            return self.prev_a.back_up_q(self.r + gamma * self.v,self.n)
        else:
            return None
class act_node:
    def __init__(self,a,s_node):
        self.a = a
        self.states = []
        self.prev_s = s_node
        self.n = 0
    def expand(self,s,r,q, p):
        for st in self.states:
            if np.equal(st.s,s).all():
                st.r = r
                return st
        st = state_node(s,self,q, p)
        st.r = r
        self.states.append(st)
        return st
    def back_up_q(self,v,sn):
        self.n += 1
        qsa = self.prev_s.q[self.a]
        qsa += (v - qsa) * sn / self.n
        self.prev_s.q[self.a] = qsa
        return self.prev_s
back_itr = 0
class state_node_determ:
    def __init__(self,s,s_node,Qsa,Ps,qmask,self_play=False,player=None):
        self.a = -1
        self.s = s
        self.self_play = self_play
        self.player = player
        self.prev_a = s_node
        self.acts = {} #act_node(i,self) for i in range(policy_len)
        self.q = np.copy(Qsa)
        self.pi = np.copy(Ps)
        self.qmask = np.copy(qmask)
        self.v = -100
        self.r = 0
        self.n = 0
    def select(self,env:Board):
        act_mask = env.act_mask()
        tp = self.pi * act_mask
        self.pi = tp/tp.sum()
        a = np.random.choice(range(policy_len),p=self.pi)
        if a not in self.acts:
            act_n = state_node_determ(-1,self,None,None,None)
            act_n.a = a
            self.acts[a] = act_n
            self.qmask[a] = 1.
        return self.acts[a],act_mask
    def expand(self,s,r,q, p,qmask,self_play=False,player=-1):
        self.s = s
        self.r = r
        self.q = np.copy(q)
        self.pi = np.copy(p)
        self.qmask = np.copy(qmask)
        self.self_play = self_play
        self.player = player
        return self
    def back_up_v(self):
        self.n += 1
        global back_itr
        back_itr += 1
        self.v = np.max((self.q + 1000)*self.qmask) - 1000 * self.qmask.any()
        if self.v >0 and np.max(self.q) < 0:
            print()
        # self.v = np.max(self.q)
        # if back_itr > 2 and self.v and max(self.acts.values(),key=lambda x: x.v).v > 0:
        #     print(f"here v={self.v}")
        # print(f"here {self.s} is {self.v}")
        # pi2 = update_Pi(self.pi,self.q,Lr)
        self.pi = update_Pi(self.pi,self.q,Lr)
        if self.prev_a:
            pa = self.prev_a
            qmax = -self.v if self.self_play and self.player != pa.player else self.v
            pa.q[self.a] = self.r + gamma * qmax
            return pa
        else:
            return None
mcts_steps = 31
stim_episode_counts = 20
class snake_net(nn.Module):
    def __init__(self,n):
        super(snake_net, self).__init__()
        self.emb = nn.Embedding(n,16,0)
        self.fc = nn.Sequential(nn.Linear(16, 16), nn.ReLU(inplace=True))
        self.fc_Q = nn.Linear(16,policy_len)
        self.fc_Pi = nn.Sequential(nn.Linear(16, policy_len), nn.ReLU(inplace=True))
    def forward(self,input):
        emb = self.emb(input)
        act_logits = self.fc(emb)
        q_logits = self.fc_Q(act_logits)
        pi_logits = self.fc_Pi(act_logits)
        return q_logits,pi_logits
# flag_print = False
def mcts_one_step_q(root_game:Board,qnet,qmap,self_play=False):
    root_state = root_game.current_state()
    # rh = root_state
    rh = root_state.view(np_hash)
    if rh in qmap:
        q, p,qmask = qmap[rh]
    else:
        # q, p = qnet(root_state)
        # p = p.softmax(dim=-1)
        q, p,qmask = np.zeros((policy_len,),dtype=float),(np.zeros((policy_len,)) + 1/policy_len),np.zeros((policy_len,),dtype=float)
    root_node = state_node_determ(root_state,None,q,p,qmask,self_play,root_game.current_player)
    """total episodes """
    for epi in range(20):
        env = copy.deepcopy(root_game)
        s_node = root_node
        for step in range(mcts_steps):
            a_node,mask = s_node.select(env)
            s,r,t,_ = env.step(a_node.a)
            if not t and step == mcts_steps - 1:
                t = True
            # sh = s
            sh = s.view(np_hash)
            if sh in qmap:
                q, p, qmask = qmap[sh]
                # if env.states:
                #     print(f"-----state\n{s} in qmap----")
            else:
                q, p, qmask = np.zeros((policy_len,), dtype=float), (
                            np.zeros((policy_len,)) + 1 / policy_len), np.zeros((policy_len,), dtype=float)
            s_node = a_node.expand(s,r,q,p,qmask,self_play,env.current_player)
            if t:
                global back_itr
                back_itr = 0
                while s_node:
                    s_node = s_node.back_up_v()
                break
    q0,p0,m0 = root_node.q,root_node.pi,root_node.qmask
    # qmap[rh] = (q0,p0,m0)
    if root_game.states:
        for i in range(4):
            rht = np.ascontiguousarray(np.rot90(rh,i))
            qt = np.rot90(q0.reshape((6, 6)),i)
            pt = np.rot90(p0.reshape((6, 6)), i)
            mt = np.rot90(m0.reshape((6, 6)), i)
            qmap[rht.view(np_hash)] = (qt.flatten(), pt.flatten(), mt.flatten())
            rhf = np.ascontiguousarray(np.fliplr(rht))
            qf = np.fliplr(qt)
            pf = np.fliplr(pt)
            mf = np.fliplr(mt)
            qmap[rhf.view(np_hash)] = (qf.flatten(), pf.flatten(), mf.flatten())
    return root_node.q,root_node.pi,root_node.qmask
def mcts_one_step(root_state):
    pass
    # if root_state >= N:
    #     print(f"state {root_state} is already end")
    #     return
    # root_node = state_node(root_state,None,Q[root_state],Pi[root_state])
    # """total stimulate game counts """
    # for epi in range(stim_episode_counts):
    #     env = myEnv(lad,policy)
    #     env.pos= root_state
    #     s_node = root_node
    #     """one game total steps """
    #     for step in range(mcts_steps):
    #         a = s_node.select()
    #         s,r,t,_ = env.step(a)
    #         if not t and step == mcts_steps - 1:
    #             t = True
    #         a_node = s_node.acts[a]
    #         s_node = a_node.expand(s,r,Q[s],Pi[s])
    #         if t:
    #             while s_node:
    #                 s_node = s_node.back_up_v()
    #             break
    # Pi[root_state] = root_node.pi
    # Q[root_state] = root_node.q
    # return root_node
def update_Pi(pi,q,lr = 0.01):
    pi = np.array(pi)
    q = np.array(q - np.min(q,axis=-1,keepdims=True)) + 10
    abc = np.log(pi+1E-10)
    dabc = q - q.sum(axis=-1,keepdims=True) * pi
    abc += dabc * lr
    z = np.exp(abc)
    pi2 = z / z.sum(axis=-1,keepdims=True)
    return pi2
def cross_entropy_test():
    lr = 0.01
    a0, b0, c0 = 0, 0, 0
    a, b, c = np.exp(a0), np.exp(b0), np.exp(c0)
    s = a + b + c
    pa = a / s
    pb = b / s
    pc = c / s
    q = [[-20, 0, 0], [-20, -20, 0], [-20, -20, 80], [60, -20, 80]]
    for i in range(len(q)):
        qa, qb, qc = q[i]
        da = (qa / pa - (qa + qb + qc)) * a / s
        db = (qb / pb - (qa + qb + qc)) * b / s
        dc = (qc / pc - (qa + qb + qc)) * c / s
        a0 += da * lr
        b0 += db * lr
        c0 += dc * lr
        a, b, c = np.exp(a0), np.exp(b0), np.exp(c0)
        s = a + b + c
        pa = a / s
        pb = b / s
        pc = c / s
        # print(a0, b0, c0)
        print(pa, pb, pc)
def tabular_games():
    root_s = 1
    # env = myEnv(lad, policy)
    # epis = []
    # """ total games """
    # for epoch in range(30):
    #     env.reset()
    #     s = root_s
    #
    #     """ total steps per game """
    #     for step in range(mcts_steps):
    #         """give me one step """
    #         root_node = mcts_one_step(s)
    #         q, pi = root_node.q,root_node.pi
    #         print(q)
    #         print(pi)
    #         tmp = q+pi
    #         a = np.argmax(tmp)
    #         s2, r, t, _ = env.step(a)
    #         if not t and step == mcts_steps - 1:
    #             t = True
    #         epis.append((s,a,t,q,pi))
    #         print(f'here we are {s2} by policy {a}')
    #         s = s2
    #         if t:
    #             PI = Q.argmax(axis=-1)
    #             print(f"we{epoch}_{step} are done!")
    #             print(PI)
    #             break
def qmap_games():
    root_s = 1
    # env = myEnv(lad, policy)
    # # qnet = snake_net(N)
    # qmap = {}
    # epis = []
    # print(f"game start! N = {N},ladder={lad},policy={policy}")
    # """ total trains """
    # for epoch in range(30):
    #     env.reset()
    #     s = root_s
    #     """ total sample steps per game """
    #     for step in range(mcts_steps):
    #         """tell me what to do on s """
    #         q, p = mcts_one_step_q(s, None, qmap)
    #         print(q)
    #         print(p)
    #         tmp = q + p
    #         a = np.argmax(tmp)
    #         s2, r, t, _ = env.step(a)
    #         if not t and step == mcts_steps - 1:
    #             t = True
    #         epis.append((s, a, q, p))
    #         s = s2
    #         print(f'here we are {s2} by policy {a}')
    #         if t:
    #             print(f"we{epoch}_{step} are done!")
    #             break
def gumoko_games():
    from learning.gumoko_env import Board
    env = Board(width=height, height=height, n_in_row=4)
    # qnet = snake_net(N)
    # qmap = {}
    qmap = np.load('d:/my_qmap.npy',allow_pickle=True).item()
    # epis = []
    print(f"game start!")
    """ total trains """
    epoch = -1
    while 1:
        epoch += 1
        start_player = np.random.randint(2)
        env.init_board(start_player)
        """ total sample steps per game """
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch} saving qmap len = {len(qmap)}")
            np.save('d:/my_qmap.npy', qmap)

        for step in range(mcts_steps):
            """tell me what to do on s """
            q, p,m = mcts_one_step_q(env, None, qmap,True)
            # print(q)
            # print(p)
            tmp = q + np.random.randn(policy_len) * 0.001
            mask = env.act_mask()
            a = np.argmax((tmp + 1000)*mask)
            s2, r, t, _ = env.step(a)
            if not t and step == mcts_steps - 1:
                t = True
            # epis.append((s, a, q, p,m))
            # s = s2
            # print(f'here we are\n{s2} by policy {a}')
            if t:
                print(f'start by {env.players[start_player]} finallly here we are\n{s2} by policy {a}')
                print(f"we{epoch}_{step} are done!")
                print(len(qmap))
                break
    return qmap
def gumoko_pk():
    from learning.gumoko_env import Board
    env = Board(width=height, height=height, n_in_row=4)
    # qnet = snake_net(N)
    # qmap = {}
    qmap = np.load('d:/my_qmap.npy',allow_pickle=True).item()
    print(f"game start!")
    """ total trains """
    for epoch in range(1):
        start_player = np.random.randint(2)
        env.init_board(start_player)
        noise = np.random.randn(policy_len) * 0.001
        s = env.current_state()
        """ total sample steps per game """
        if (epoch + 1) % 100 == 0:
            np.save('d:/my_qmap.npy', qmap)
        a = -1
        for step in range(mcts_steps):
            print(f'here we are\n{s} by policy {a}')
            if env.current_player == 1:
                move = -1
                while move < 0:
                    try:
                        location = input("Your move: ")
                        if isinstance(location, str):  # for python3
                            location = [int(n, 10) for n in location.split(",")]
                        move = env.location_to_move(location)
                    except Exception as e:
                        move = -1
                    if move == -1 or move not in env.availables:
                        print("invalid move")
                        move = -1
                a = move
            else:
                """tell me what to do on s """
                q, p,m = mcts_one_step_q(env, None, qmap,True)
                # print(q)
                # print(p)
                # if step > mcts_steps - 5:
                #     print()
                tmp = q + noise
                mask = env.act_mask()
                a = np.argmax((tmp + 1000)*mask)
            s2, r, t, _ = env.step(a)
            if not t and step == mcts_steps - 1:
                t = True
            # epis.append((s, a, q, p,m))
            s = s2
            if t:
                print(f'start by {env.players[start_player]} finallly here we are\n{s2} by policy {a}')
                print(f"we{epoch}_{step} are done!")
                print(len(qmap))
                break
    return qmap
M=0
if __name__ == '__main__':
    # qmap = gumoko_pk()
    qmap = gumoko_games()
    # np.save('d:/my_qmap.npy', qmap)
    # tabular_games()
#
# p = np.array([[0.45,0.55],[0.45,0.55],[0.45,0.55],[0.45,0.55]])
# # q = np.array([[-20, 0], [-10, 10], [0, 20], [10,30]])
# q = np.array([[-20, -20], [-10, -10], [1, 1],[20, 20]])
# for i in range(10):
#     print("----")
#     print(p)
#     for k in range(len(q)):
#         pa = update_Pi(p,q,0.01)
#         p = pa
# print(p)

