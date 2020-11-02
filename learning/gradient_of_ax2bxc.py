import torch
import numpy as np
x = torch.arange(-100,100)

a = torch.tensor(0.05,requires_grad=True)
b = torch.tensor(-0.03,requires_grad=True)
c = torch.tensor(-0.02,requires_grad=True)
def hk(gr):
        print("grad=",gr)
a.register_hook(hk),b.register_hook(hk),c.register_hook(hk)
opt = torch.optim.Adam([a,b,c],lr=0.01)
for i in range(3):
    xx = torch.tensor(np.random.choice(x,32))
    # xx = torch.tensor([0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4])
    yy = 3 * xx * xx + 8 * xx + 5
    opt.zero_grad()
    loss = a*xx*xx + b*xx + c - yy
    loss = loss * loss
    loss = loss.mean()
    loss.backward()
    opt.step()
    print(f"{a}  {b}  {c}")