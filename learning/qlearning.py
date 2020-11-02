import numpy as np
GAMMA = 0.8
ss = 2
aa = 2
Q = np.zeros((ss,aa))
S2 = np.zeros((ss,aa))
# R=np.asarray([[-1,-1,-1,-1,0,-1],
#    [-1,-1,-1,0,-1,100],
#    [-1,-1,-1,0,-1,-1],
#    [-1,0, 0, -1,0,-1],
#    [0,-1,-1,0,-1,100],
#    [-1,0,-1,-1,0,100]])
R=np.asarray([[0,1],
   [-1,-1]])
S2=np.asarray([[0,1],
   [0,1]])
pi = [1,1]
def getMaxQ(s,a):
    return max(Q[S2[s,a], :])
def QLearning(state):
    for action in range(aa):
        if(R[state][action] == -1):
            Q[state, action]=0
        else:
            Q[state,action]=R[state][action]+GAMMA * getMaxQ(state, action)
def QLearning2(state,action):
    Q[state,action]=R[state][action]+GAMMA * getMaxQ(state, action)
def QLearning3(state,action):
    Q[state,action]=R[state][action]+GAMMA * Q[S2[state,action],pi[S2[state,action]]]
def policy_imp():
    return np.argmax(Q,axis=-1)
count=0
while count<1000:
    QLearning2(0,1)
    QLearning2(1,0)
    '''learn to fail is also very important'''
    # QLearning2(1,1)
    pi = policy_imp()
    count+=1
print(Q)