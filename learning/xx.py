import gym
import numpy as np
from collections import defaultdict

m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
n = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]

print('list(zip(m,n)):\n', list(zip(m, n)))
print("*zip(m, n):\n", *zip(m, n))
print("*zip(*zip(m, n)):\n", *zip(*zip(m, n)))

m2, n2 = zip(*zip(m, n))
print(m == list(m2) and n == list(n2))
