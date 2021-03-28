import gym
import numpy as np
from collections import defaultdict
# a = defaultdict(int)
# for i in range(10):
#     a[(1+i,2+i,3+i)] = i
# k = a.keys()
# lk = list(k)
# rk = range(len(k))
# idx = np.random.choice(rk,11)

# if not hasattr(hasattr, '__hash'):
#     self.__hash = int(sha1(self.view(uint8)).hexdigest(), 16)
# return self.__hash
# b = np.random.choice(list(range(len(a))),p=a/a.sum())

b = {}
a = [[0,5],[1,3],[0,3],[1,2],[1,1]]
a1 = [[6,5],[1,3],[2,3],[2,2],[1,1]]
aa = []
i = 0
while i != len(a):
    print(i)
    st = a[i]
    if st[0] in b:
        a[b[st[0]]][1] += st[1]
        a.pop(i)
        print("pop")
    else:
        b[st[0]] = i
        i += 1
print(a)
c = a + a1
print(c)
a1.sort()
print(c)
a1[2][1] = 9
print(c)
a1.pop(4)
print(c)
