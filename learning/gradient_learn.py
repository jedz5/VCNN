import torch as t
from torch.autograd import Variable as v
from torch.optim import SGD
import torch.nn.functional as F

def xx():
    #梯度与样本量成比例
    a = t.tensor([0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
    opt = SGD([a], lr=0.1)
    # loss = .5 * (a - 1)**2
    loss = .5 * F.mse_loss(a, t.tensor([1., 1., 1., 1., 1., 1., 1., 1.]), reduction='sum')  # .sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a)
# simple gradient
a = v(t.FloatTensor([2, 3]), requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()
print('*'*10)
print('=====simple gradient======')
print('input')
print(a.data)
print('compute result is')
print(out.data)
print('input gradients are')
print(a.grad.data)
out.backward()
print('input gradients are')
print(a.grad.data)

# backward on non-scalar output
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
n = v(t.zeros(1, 2))
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
n.backward(t.FloatTensor([[1, 1]]))
print('*'*10)
print('=====non scalar output======')
print('input')
print(m.data)
print('input gradients are')
print(m.grad.data)


# jacobian
j = t.zeros(2 ,2)
k = v(t.zeros(1, 2))
m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
# [1, 0] dk0/dm0, dk1/dm0
k.backward(t.FloatTensor([[1, 0]]), retain_variables=True) # 需要两次反向求导
j[:, 0] = m.grad.data
m.grad.data.zero_()
# [0, 1] dk0/dm1, dk1/dm1
k.backward(t.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)