import numpy as np
from matplotlib import pyplot as plt
round = 10
num_ep=80
K=32
ps = []
for it in range(round):
    e1 = 1400
    e2 = 1400
    r = np.random.binomial(1,0.76,num_ep)
    e1s =[e1]
    e2s =[e2]
    es = [0]
    for ir in r:
        exp1 = 1/(1+10**((e2 - e1)/400))
        points = int(K*(ir-exp1))
        e2 = e2 - points
        e1 = e1 + points
        e1s.append(e1)
        e2s.append(e2)
        # print("e1",e1)
        # print("e2",e2)
    mes  = np.mean(e1s[-(num_ep-20):]) - np.mean(e2s[-(num_ep-20):])
    P = 1/(1+10**((-mes)/400))
    ps.append(P)
    print(P)
print("P",np.mean(ps))
xs = np.arange(0,len(r)+1,1)
plt.scatter(xs,e1s,marker='x')
plt.scatter(xs,e2s,marker='*')
plt.show()