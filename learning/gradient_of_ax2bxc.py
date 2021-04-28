import torch
import numpy as np
import torch.nn.functional as F
torch.set_printoptions(precision=3)
x = torch.arange(-10,10,dtype=float)

param = torch.tensor([0.,0.,0.],requires_grad=True)
def hk(gr):
        print("梯度=",gr)
param.register_hook(hk)
lrate = 0.2
opt = torch.optim.Adam([param],lr=lrate)
for i in range(3):
    xx = torch.tensor(np.random.choice(x,32))
    param_before = param.clone()
    yy = 3 * xx * xx + 8 * xx + 5
    opt.zero_grad()
    y_pred = param[0]*xx*xx + param[1]*xx + param[2]
    loss = F.smooth_l1_loss(y_pred,yy)
    loss.backward()
    opt.step()
    # print(f"更新[{param_before[0]} {param_before[1]} {param_before[2]}] + {lrate}*grad = {param[0]}  {param[1]}  {param[2]}")
    print(f"更新{param_before} - {lrate}*grad = {param}")