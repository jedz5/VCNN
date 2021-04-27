import torch
import torch.nn.functional as F
from torch import nn
p = torch.tensor([0.2,0.3,0.5])

# a = torch.tensor(1,requires_grad=True)
# b = torch.tensor(1,requires_grad=True)
# c = torch.tensor(1,requires_grad=True)

# p1 = torch.tensor([0.5,0.3,0.7],requires_grad=True)
# op = torch.optim.Adam([p1],0.1)
# for i in range(100):
#     # loss = torch.nn.functional.cross_entropy(p,p1)
#     #只能对pred计算log 不能对label算log
#     loss = - (p1.softmax(dim=-1) * p.log()).sum()
#     op.zero_grad()
#     loss.backward()
#     op.step()
#     print(p1.softmax(dim=-1))

'''单样本不能使用adam？？'''

'''batch constrained Q learning'''
p1 = torch.tensor([[0.,0.,0.]],requires_grad=True)
label_imt = torch.tensor([[1],[2],[1],[0],[1],[0],[1],[0]])
# label_imt = torch.tensor([[1,0,0],[0,1,0],[1,0,0],[1,0,0]])
# p1 = torch.tensor([0.1,0.1,0.1],requires_grad=True)
op = torch.optim.Adam([p1],lr=0.0001)
op.zero_grad()
op.step()
def hook_me(grad):
    print("grad ",grad)
# p1.register_hook(hook_me)
for i in range(len(label_imt)):
    p_sigmod = p1.sigmoid()
    loss = F.nll_loss(p_sigmod,label_imt[i])
    # loss = - (label_imt[i] * p_sigmod.log()).sum()
    op.zero_grad()
    loss.backward()
    op.step()
    print(p1.sigmoid() > 0.5)