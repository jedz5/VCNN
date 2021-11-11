import torch as t
from torch.autograd import Variable as v
from torch.optim import SGD
import torch.nn.functional as F


r'''梯度与样本量成比例'''
def xx():
    a = t.tensor([0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
    opt = SGD([a], lr=0.1)
    # loss = .5 * (a - 1)**2
    loss = .5 * F.mse_loss(a, t.tensor([1., 1., 1., 1., 1., 1., 1., 1.]), reduction='sum')  # .sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a)
r'''mask 对 value loss的影响'''
def xx2():
    a = t.tensor([1.,1.],requires_grad=True)
    a_ind = a.gather(0,t.tensor([0,0],dtype=t.long))
    b = t.tensor([2.,2.])
    mask = t.tensor([1., 1.])
    opt = SGD([a], lr=0.1)
    loss = F.smooth_l1_loss(mask*a_ind,b,reduction='sum')
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a)

if __name__ == '__main__':
    xx2()