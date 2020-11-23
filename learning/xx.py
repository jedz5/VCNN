import numpy as np
import torch
from torch import nn
from scipy import linalg


a = nn.Linear(14 + 16, 64)
aa = []
a1 = torch.tensor(np.arange(64*7*140).reshape(64,7,140))
mask = torch.ones((64,7))
a1 = a1 * mask.unsqueeze(-1)
emb = nn.Embedding(10,16)
c = emb(torch.tensor([1,1,1]))
index = torch.tensor((np.arange(64*1).reshape((64,1)) % 7),dtype=torch.long)
# index[:,:,2] = 3
# c1 = a1.gather(1,index)
# c2 = emb(torch.tensor([2]))
# c = b * a1

ind = index.view(64,1,1).expand((64,1,140))
c1 = a1.gather(1,ind,)