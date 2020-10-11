import torch

p = torch.tensor([0.2,0.3,0.5])

# a = torch.tensor(1,requires_grad=True)
# b = torch.tensor(1,requires_grad=True)
# c = torch.tensor(1,requires_grad=True)

p1 = torch.tensor([0.5,0.3,0.7],requires_grad=True)
op = torch.optim.Adam([p1],0.1)
for i in range(100):
    # loss = torch.nn.functional.cross_entropy(p,p1)
    #只能对pred计算log 不能对label算log
    loss = - (p1.softmax(dim=-1) * p.log()).sum()
    op.zero_grad()
    loss.backward()
    op.step()
    print(p1.softmax(dim=-1))