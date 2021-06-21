import numpy as np
np.set_printoptions(precision=1,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.3f}'.format})
GAMMA = 0.8
ss = 2
aa = 2
Q = np.zeros((ss,aa))
# Q=np.asarray([[0.4,0.5],
#    [-.5,-1.4]])
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
def getMaxQ(s,a):
    return max(Q[S2[s,a], :])
def QLearning(state):
    for action in range(aa):
        if(R[state][action] == -1):
            Q[state, action]=0
        else:
            Q[state,action]=R[state][action]+GAMMA * getMaxQ(state, action)
def QLearning2(pi):
    for s in range(2):
        for a in range(2):
            Q[s,a]=R[s,a]+GAMMA * getMaxQ(s,a)
def QLearning3(pi):
    for s in range(2):
        for a in range(2):
            Q[s,a]=R[s,a]+GAMMA * Q[S2[s,a],pi[S2[s,a]]]
def policy_imp():
    return np.argmax(Q,axis=-1)
count=0
pi = [0,0]
while count<1000:
    QLearning3(pi)
    # pi = policy_imp()
    count+=1
print(Q)