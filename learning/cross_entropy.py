import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.4f}'.format})
torch.set_printoptions(precision=4,sci_mode=False,linewidth=800)
'''交叉熵 p1 logp2作图'''
# fig=plt.figure()
# ax=Axes3D(fig)
# X=np.linspace(0.01,0.99,101)
# Y=np.linspace(0.01,0.99,101)
# X,Y=np.meshgrid(X,Y)
# Z=-(X*np.log2(Y)+(1-X)*np.log2(1-Y))
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
# plt.show()
def get_cross_entropy(r,point_n = 100):
    assert len(r) == 3
    z = np.zeros((point_n,point_n)) - 5
    for y in range(1,point_n):
        x = np.array(range(1,point_n-y))
        z_real= -(r[0]*np.log(x/point_n)+r[1]*np.log((point_n-y-x)/point_n)+r[2]*np.log(y/point_n))
        z[y, 1:point_n - y] = z_real
    return z
'''交叉熵 2logx + 3logy -logz,x+y+z =1'''
# fig=plt.figure()nb0
# ax=Axes3D(fig)
# z = get_cross_entropy()
# X,Y=np.meshgrid(np.array(range(point_n))/point_n,np.array(range(point_n))/point_n)
# ax.plot_surface(X,Y,z,cmap='rainbow')
# plt.show()
'''交叉熵 固定p 只对logp作图'''
# fig=plt.figure()
# X=0.8
# Y=np.linspace(0.01,0.99,101)
# # Z=-(X*np.log2(Y)+(1-X)*np.log2(1-Y))
# Z=-(5*np.log2(Y)-3*np.log2(1-Y))
# plt.plot(Y, Z)
# plt.show()
'''不论函数f是增函数还是减函数，梯度△=df/dx始终代表函数值增长的方向
梯度下降法 -= df/dx,是求函数的局部最小值,与函数增减性无关
梯度上升法 += df/dx,是求函数的局部最大值,与函数增减性无关
梯度上升法,其几何意义和很好理解，那就是：算法的迭代过程是一个“上坡”的过程，
每一步选择坡度变化率最大的方向往上走，这个方向就是函数在这一点梯度方向（注意不是反方向了）。最后随着迭代的进行，梯度还是不断减小，最后趋于0。
所谓的梯度“上升”和“下降”，一方面指的是你要计算的结果是函数的极大值还是极小值。计算极小值，就用梯度下降，计算极大值，就是梯度上升；
另一方面，运用上升法的时候参数是不断增加的，下降法是参数是不断减小的。但是，在这个过程中，“梯度”本身都是下降的，直到趋于0。
'''

# r = torch.tensor([1,1,1,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],dtype=torch.float32)
# n = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=torch.float32)
# logits = torch.tensor([1.5,1.5,1.5,1,1,1,1,1,1,1,1,1,1,1],requires_grad=True,dtype=torch.float32)

# r = torch.tensor([10 , 1 , -1 , -1, -1 , -1])
# p = torch.tensor([0.01 , 1 , 1 , 1, 1 , 1])
# logits = torch.tensor([1,1,1,1,3,1],requires_grad=True,dtype=torch.float32)
# r = torch.tensor([4,2 , 1,2,2,2,2],dtype=torch.float32)
# n = torch.tensor([1, 1 , 1, 1, 1, 1, 1],dtype=torch.float32)
# logits = torch.tensor([1,1,1,1,1,1,1],requires_grad=True,dtype=torch.float32)
'''交叉熵 梯度计算 梯度下降大头在负概率部分 正概率对梯度贡献小
    pred_p = logits.softmax()
    loss = -sum(r * p * log(pred_p))
    △= d(loss)/d(logits[i]) = -∑j [rj* pj * d(loss)/d(log(pred_pj)) * d(log(pred_pj))/d(logitsi)]
                             = r_mean*pred_pi - ri*pi
    △= d(loss)/d(logits[1]) = -d(r1*p1*log(pred_p1) + r2*p2*log(pred_p2) + r3*p3*log(pred_p3)) / d(logits1)
                             = -(r1*p1*(1/pred_p1)*pred_p1(1-pred_p1) + r2*p2(1/pred_p2)*(pred_p2*(0-pred_p1) + r3*p3(1/pred_p3)*(pred_p3*(0-pred_p1))
                             = -(r1*p1 - (p1*r1+p2*r2+p3*r3)*pred_p1)
                             = r_mean*pred_p1 - r1*p1
    ri = r[r>0],rj = r[r<0] ,δ=sum(rj*pj),N=len(ri)
    target = (ri*pi+δ/N)/r_mean
    对任意ri*pi > r_mean*pred_pi,△ < 0 ,logits增大
        当 r_mean > 0,则ri>0
            若存在i,使得targeti > 1,则idx = argmax(target), pred_p[idx] 趋近1,其余pred_p趋近0  eg. r = [6,0.9,-3.9] n = [1,1,1]
            若任意i,均有targeti < 1,则pred_pi趋近targeti, pred_pj趋近0,  eg. r = [3,3.9,-0.9] n = [1,1,1]
        当 r_mean < 0
            当ri > 0,则ri*pi/r_mean < 0, 则idx = f(pred_p,r*p), pred_p[idx] 趋近1,其余pred_p趋近0 f未知...
                                             eg. r = [1.2,1,-3.9,-0.6] n = [1,2,1,1] logits = [5.,[3.4,3.5],3.5,3.5] lr =[0.2,0.3]
                                                 r = [1.2, 1,-1] n = [1,  [2,4],10]  logits = [1.3,1,.3]
    当 r_mean > 0
        ri > 0
            若pred_pi > ri*pi/r_mean 有 △ = r_mean*pred_pi - ri*pi > 0, 则 logitsi 减小,pred_pi减小,pred_pi 趋近 ri*pi/r_mean
            若pred_pi < ri*pi/r_mean 有 △ = r_mean*pred_pi - ri*pi < 0, 则 logitsi 增大
                若1 < ri*pi/r_mean,
        ri < 0,则△ = r_mean*pred_pi - ri*pi > 0 始终成立,则 logitsi 减小,pred_pi 趋近 0
    当 r_mean < 0
        ri < 0
            若pred_pi > ri*pi/r_mean 有 △ = r_mean*pred_pi - ri*pi < 0, 则 logitsi 增大,pred_pi 趋近 1
            若pred_pi < ri*pi/r_mean 有 △ = r_mean*pred_pi - ri*pi > 0, 则 logitsi 减小,pred_pi 趋近 0
        ri > 0,则△ = r_mean*pred_pi - ri*pi < 0 始终成立,则 logitsi 增大,pred_pi 趋近 1
    
'''
'''
Adm虽然能得到使得loss更小的梯度
logits=[ 20.6662  20.4484  20.4484 -18.6501]	pred_p=[ 0.3834  0.3083  0.3083  0.0000]	loss=[ 0.9588  0.2942  0.2942 -35.2408] loss.sum=-33.693672
但SGD能得到使结果更准确的梯度
logits=[ 10.1542  5.1997  5.1997 -16.5534]	pred_p=[ 0.9861  0.0070  0.0070  0.0000]	loss=[ 0.0140  1.2421  1.2421 -23.3814] loss.sum=-20.883131
'''
r = np.array([-1.2,-.9,-1],dtype=float)
n = np.array([100,  100,100],dtype=float)
logits = np.array([2,1,1],dtype=float) #np.array([4,3.4,3.5,3.5],dtype=float)
p = n / n.sum()
r = r * p
neg_rp = sum(r[r < 0])
pos_N = len(r[r > 0])
from scipy.special import softmax
lr = 0.2
pred_p = softmax(logits,axis=-1)
pred_p_init = np.copy(pred_p)
r_mean = sum(r)
'''1/(1+(r2*p2+r3*p3)/r1*p1)'''
target = r / r_mean
target_2 = (r + neg_rp/pos_N) / r_mean
'''(r2*p2+r3*p3)/r1*p1'''
ratio = (r_mean - r)/r
grads_l = []
logits_l = []
preds_l = []
N = 500
for i in range(N):
    logits_l.append(np.copy(logits))
    print(logits, end="\t")
    preds_l.append(np.copy(pred_p))
    print(pred_p,end="\t")
    loss =  -r * np.log(pred_p)
    print(loss, end="\t")
    loss = loss.sum()
    print(loss)
    grad = r_mean*pred_p - r
    grads_l.append(np.copy(grad))
    logits = logits - lr*grad
    print("grad ", grad, end="\t")
    pred_p = softmax(logits-logits.max(),axis=-1)
print()
print(f'r_mean      ={r_mean}')
print(f'r*p         ={r}')
print(f'r_mean*p_   ={pred_p_init*r_mean}')
print(f'p           ={p}')
print(f'pred_p_init ={pred_p_init}')
print(f'target      ={target}')
print(f'target_2    ={target_2}')
fig=plt.figure()
x = np.array(list(range(N)))
grads_l = np.array(grads_l).transpose()
logits_l = np.array(logits_l).transpose()
preds_l = np.array(preds_l).transpose()
plt.subplot(3, 1, 1) #3行1列子图，当前画在第一行第一列图上
for i in range(len(grads_l)):
    plt.plot(x,grads_l[i])
plt.subplot(3, 1, 2)
for i in range(len(logits_l)):
    plt.plot(x,logits_l[i])
plt.subplot(3, 1, 3)
for i in range(len(preds_l)):
    plt.plot(x,preds_l[i])
point_n = N
z = get_cross_entropy(r,point_n) + 10
xline = preds_l[0]*(point_n-1)
yline = preds_l[2]*(point_n-1)
zline = z[yline.astype(int),xline.astype(int)]
X,Y=np.meshgrid(np.array(range(point_n))/point_n,np.array(range(point_n))/point_n)
fig2=plt.figure()
ax=Axes3D(fig2)
ax.plot_surface(X,Y,z,cmap='rainbow',alpha=0.5)
ax.scatter3D(preds_l[0], preds_l[2], zline,color='black')
plt.show()
'''log函数下降的速度并没有想象中快啊
-np.log(0.1)
2.3025850929940455
-np.log(0.00001)
11.512925464970229
-np.log(0.000001)
13.815510557964274
-np.log(0.0000001)
16.11809565095832
-np.log(0.00000001)
18.420680743952367
'''
# a = torch.tensor([0.2,0.8,-0.3])
# b = torch.tensor([0.3,0.7,-0.3])
# c = torch.tensor([0.4,0.6,-0.3])
# p = torch.tensor([0.4,0.6,0.00001])
# loss = -a*p.log()

'''batch constrained Q learning'''
# p1 = torch.tensor([[0.,0.,0.]],requires_grad=True)
# label_imt = torch.tensor([[1],[2],[1],[0],[1],[0],[1],[0]])
# # label_imt = torch.tensor([[1,0,0],[0,1,0],[1,0,0],[1,0,0]])
# # p1 = torch.tensor([0.1,0.1,0.1],requires_grad=True)
# op = torch.optim.Adam([p1],lr=0.0001)
# op.zero_grad()
# op.step()
# def hook_me(grad):
#     print("grad ",grad)
# # p1.register_hook(hook_me)
# for i in range(len(label_imt)):
#     p_sigmod = p1.sigmoid()
#     loss = F.nll_loss(p_sigmod,label_imt[i])
#     # loss = - (label_imt[i] * p_sigmod.log()).sum()
#     op.zero_grad()
#     loss.backward()
#     op.step()
#     print(p1.sigmoid() > 0.5)