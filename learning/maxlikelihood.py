import torch
from torch import tensor
import numpy as np
sample_n1 = 100
sample_n2 = 200
num_it = 2500
r_mu0 = 5
r_mu1 = 7
r_sig0 = 0.5
r_sig1 = 0.7
np.random.seed(123)
type = torch.float32
Z = tensor(1,requires_grad = True,dtype=type)
mu0 = tensor(1,requires_grad = True,dtype=type)
sigma0 = tensor(np.log(0.2),requires_grad = True,dtype=type)
mu1 = tensor(1.2,requires_grad = True,dtype=type)
sigma1 = tensor(np.log(0.3),requires_grad = True,dtype=type)


x0 = np.random.normal(r_mu0,r_sig0,sample_n1)
x1 = np.random.normal(r_mu1,r_sig1,sample_n2)
xs = []
xs.extend(x0)
xs.extend(x1)
X = tensor(xs,requires_grad = False,dtype=type)
optimizer = torch.optim.Adam([Z,mu0,mu1,sigma0,sigma1],lr=0.01)
print(torch.sigmoid(Z))
print(mu0)
print(torch.exp(sigma0))
print(mu1)
print(torch.exp(sigma1))
losses = []
for i in range(num_it):
    optimizer.zero_grad()
    sig0 = torch.exp(sigma0)+ 1E-10
    sig1 = torch.exp(sigma1)+ 1E-10
    z = torch.sigmoid(Z)
    n0 = torch.exp(-0.5*((X - mu0)/sig0)**2)/(np.sqrt(2*np.pi) * sig0) + 1E-10
    n1 = torch.exp(-0.5*((X - mu1)/sig1)**2)/(np.sqrt(2*np.pi) * sig1) + 1E-10
    total = -torch.log((z*n0 + (1-z)*n1)).sum()/(sample_n1+sample_n2)
    total.backward()
    losses.append(-total)
    optimizer.step()
print(torch.sigmoid(Z))
print(mu0)
print(torch.exp(sigma0))
print(mu1)
print(torch.exp(sigma1))
# normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
u1 = np.arange(0, 12, 0.02)
# x数对应的概率密度
rz = sample_n1/(sample_n1 + sample_n2)
y1 = normfun(u1, r_mu0,r_sig0) + 1E-10
y2 = normfun(u1, r_mu1, r_sig1)+ 1E-10
y0 = rz*y1 + (1-rz)*y2
y3 = normfun(u1, (float)(mu0.detach().numpy()),np.exp((float)(sigma0.detach().numpy())))
y4 = normfun(u1, (float)(mu1.detach().numpy()), np.exp((float)(sigma1.detach().numpy())))
pz= (float)(torch.sigmoid(Z).detach().numpy())
y5 = pz*y3 + (1-pz)*y4
from matplotlib import pyplot as plt
plt.subplot(211)
# 参数,颜色，线宽
plt.plot(u1, y0, color='black', linewidth=1)
plt.plot(u1, y1, color='g', linewidth=3)
plt.plot(u1, y2, color='r', linewidth=3)
plt.plot(u1, y5,  linewidth=2)
plt.plot(u1, y3,  color='g',linewidth=2)
plt.plot(u1, y4,  color='r',linewidth=2)
plt.scatter(x0,[1]*sample_n1,marker='x')
plt.scatter(x1, [1.2]*sample_n2,marker='x')

plt.subplot(212)
plt.plot(np.arange(0,num_it,1),losses)
plt.show()
