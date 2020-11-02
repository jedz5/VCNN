import numpy as np
from learning.my_tool import np_hash, Timer

l = []
l2 = []
with Timer("expri11") as t:
    for i in range(100):
        x = np.zeros((20, 40,60), dtype=int) + i
        l.append((x,np.zeros((16,), dtype=int) + i))
        l2.append(x.copy())
with Timer("expri12") as t:
    for a in l2:
        for x,v in l:
            if (a == x).all():
                # print("get")
                continue
l = dict()
l2 = []
with Timer("expri21") as t:
    for i in range(100):
        x = np.zeros((20, 40,60), dtype=int) + i
        l[x.view(np_hash)] = np.zeros((16,), dtype=int) + i
        l2.append(x)
with Timer("expri22") as t:
    for a in l2:
        ah = a.view(np_hash)
        if ah in l:
            b = l[ah]
            # print("found")
            continue