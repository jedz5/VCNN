import numpy as np
from learning.my_tool import np_hash, hashable
from collections import defaultdict
l = defaultdict(int)
l2 = []
a = np.zeros([2,3,4])+1

l[(tuple(a.flatten()),1)] = 22
a[1,2,3] = 5
l[(tuple(a.flatten()),1)] = 23
# l[(a.view(np_hash),1)] = 24
l[(hashable(a),1)] = 24
a[1,2,3] = 4
l[(hashable(a),1)] = 25
for i in l.keys():
    print(i)