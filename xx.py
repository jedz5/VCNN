import numpy as np
import random
from collections import deque
a = deque(maxlen=100)
a.append(1)
a.append(2)
a.append(3)
a.append(4)
a.append(5)
b = random.sample(list(a)[-2:],2)
print(b)