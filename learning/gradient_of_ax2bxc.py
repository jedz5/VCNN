import torch
import numpy as np
import torch.nn.functional as F
torch.set_printoptions(precision=3)
# x = torch.arange(-10,10,dtype=float)
xx = torch.tensor([1,2,3])
param = torch.tensor([1.,0.,0.],requires_grad=True)
def hk(gr):
        print("backward 梯度=",gr)
param.register_hook(hk)
lrate = 1
opt = torch.optim.SGD([param],lr=lrate)
for i in range(30):
    # xx = torch.tensor(np.random.choice(x,32))
    param_before = param.clone()
    yy = 3 * xx * xx + 0 * xx + 0

    #更新1次梯度
    opt.zero_grad()
    y_pred = param[0] * xx * xx + param[1] * xx + param[2]
    loss = -sum((y_pred - yy) ** 2) / 6
    loss.backward()
    print('正则化之前',param.grad)
    torch.nn.utils.clip_grad_norm_([param], 1)
    print('正则化之后', param.grad)
    # 累计4次梯度
    opt.zero_grad()
    for i in range(4):
        y_pred = param[0]*xx*xx + param[1]*xx + param[2]
        loss = -sum((y_pred-yy)**2)/6
        loss.backward()
    print('正则化之前', param.grad)
    torch.nn.utils.clip_grad_norm_([param], 1)
    print('正则化之后', param.grad)
    opt.step()
    # print(f"更新[{param_before[0]} {param_before[1]} {param_before[2]}] + {lrate}*grad = {param[0]}  {param[1]}  {param[2]}")
    print(f"更新{param_before} - {lrate}*grad = {param}")
