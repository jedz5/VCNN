import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F


class BH:
    def __init__(self,x,y,z,a,b,c):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
hh = 100
ww = 200
l2 = []
ones = np.array([BH(i%ww,(i**2+3)%hh,2,3,4,5) for i in range(70000)])
st = time.time()
oney = np.array([bh.y for bh in ones])
onex =np.array([bh.x for bh in ones])
onez =np.array([bh.z for bh in ones])
onea =np.array([bh.a for bh in ones])
oneb =np.array([bh.b for bh in ones])
onec =np.array([bh.c for bh in ones])
print(time.time() - st )
st = time.time()
onesxy = np.array([[bh.y,bh.x,bh.z,bh.a,bh.b,bh.c] for bh in ones])
oney,onex,onez,onea,oneb,onec = onesxy.transpose() #
print(time.time() - st )