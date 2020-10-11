import torch
import numpy as np
x = torch.arange(-100,100)

a = torch.tensor(0.05,requires_grad=True)
b = torch.tensor(-0.03,requires_grad=True)
c = torch.tensor(-0.02,requires_grad=True)
opt = torch.optim.Adam([a,b,c],lr=1)
for i in range(100):
    xx = torch.tensor(np.random.choice(x,32))
    yy = 3 * xx * xx + 8 * xx + 5
    opt.zero_grad()
    loss = a*xx*xx + b*xx + c - yy
    loss = loss * loss
    loss = loss.mean()
    loss.backward()
    opt.step()
    print(f"{a}  {b}  {c}")