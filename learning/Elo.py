import numpy as np
from matplotlib import pyplot as plt

a = [1,2,4,20]
b = [1,2,4,200]
X = np.array([1/16,1/8,1/4,1/3,1/2])
def hh():
    plt.figure(figsize=(8, 5))
    sz = 100
    x1 = np.sort(np.random.normal(1.7,0.1,sz))
    x2 = np.sort(np.random.normal(1.6,0.1,sz))
    # normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
    def normfun(x, mu, sigma):
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    # x的范围为60-150，以1为单位,需x根据范围调试
    u1 = np.arange(1.3, 2.0, 0.02)
    # x数对应的概率密度
    y1 = normfun(u1, 1.7,0.1)
    y2 = normfun(u1, 1.55, 0.1)
    # 参数,颜色，线宽
    plt.plot(u1, y1, color='g', linewidth=3)
    plt.plot(u1, y2, color='r', linewidth=3)
    plt.scatter(x1,[1]*sz,marker='x')
    plt.scatter(x2, [1.2]*sz,marker='x')
    print(x1)
    print(x2)
    plt.show()
def hh2():
    # plt.figure(figsize=(8,5))
    b = np.linspace(1, 6, 100)
    b_true=2
    b_start = 3.5
    for i in range(0,100):
        x0 = np.random.randint(-100,100)
        y0 = (x0 - b_true)**2#((x0-b)**2-y0)**2
        r1 = 2*x0 - b_true
        z = np.poly1d([-1,x0])
        z = z*z - [y0]
        z = z * z
        dz = z.deriv()
        dzb = dz(b_start)
        if b_start > x0:
            print("2--{}-{}-{}".format(x0,b_start,r1))
        else:
            print("2-{}--{}-{}".format(b_start,x0, r1))
        b_start -= 0.001 * dzb
        print("b = {}, dzb = {}".format(b_start,dzb))
        print("")
def entrop(p):
    p = np.array(p) / sum(p)
    x = -np.log2(p) #[log2(1 / i) for i in p]
    h = - p * np.log2(p) #[-i * log2(i) for i in p]
    return x,h


round = 10
num_ep=80
K=32
ps = []
for it in range(round):
    e1 = 1400
    e2 = 1400
    r = np.random.binomial(1,0.76,num_ep)
    e1s =[e1]
    e2s =[e2]
    es = [0]
    for ir in r:
        exp1 = 1/(1+10**((e2 - e1)/400))
        points = int(K*(ir-exp1))
        e2 = e2 - points
        e1 = e1 + points
        e1s.append(e1)
        e2s.append(e2)
        # print("e1",e1)
        # print("e2",e2)
    mes  = np.mean(e1s[-(num_ep-20):]) - np.mean(e2s[-(num_ep-20):])
    P = 1/(1+10**((-mes)/400))
    ps.append(P)
    print(P)
print("P",np.mean(ps))
xs = np.arange(0,len(r)+1,1)
plt.scatter(xs,e1s,marker='x')
plt.scatter(xs,e2s,marker='*')
plt.show()