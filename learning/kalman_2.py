import numpy as np

# 白 红 红 白 红
# x ->每次向右运动
Z = [0, 0, 1] # 观测   0：红，1：白
#P(x|x)动力方程 每次有0.1的概率前进1步，0.8的概率2步，0.1的概率3步
Pxx = np.array([[0,.1,.8,.1,0],[0,0,.1,.8,.1],[.1,0,0,.1,.8],[.8,.1,0,0,.1],[.1,.8,.1,0,0]]).T
#初始位置概率 初始先验P(x|z0) -> P(x|z1,z0) 后验
Pxz0 = np.array([[1.,0,0,0,0]]).T
#P(x|z) 先验
Pxz = np.array([[0,1,1,0,1],[1,0,0,1,0]]).T #没有看错的情况下,看到红色的墙，就应该在1,2,4位置
#P(z|x) 似然 or 观测方程
Pzx = np.array([[0,1,1,0,1],[1,0,0,1,0]]) #没有看错情况下，在0位置只能看到白色墙
#P(z|z) 误差
# Pzz = np.array([[.7,.3],[.1,.9]]).T #有0.3的概率把红的看成白的,0.1的概率把白的看成红的
# Pxz = np.matmul(Pxz ,Pzz).T
# Pzx = np.matmul(Pzz ,Pzx)
Pxzz = Pxz0
print("p(z|x)*p(x)视角<-----------------------")
for z in Z:
    Pxzz = np.matmul(Pxx ,Pxzz)
    print()
    print("Px_先验=", Pxzz.T)
    Px_pos = Pzx[z,None].T*Pxzz  #argmax(x) P(z|x) * P(x)
    Pxzz = Px_pos / sum(Px_pos)
    print("Px_后验=",Pxzz.T)

Pxzz = Pxz0
print("p(x|z)*p(x)视角<-----------------------")
for z in Z:
    Pxzz = np.matmul(Pxx ,Pxzz)
    print()
    print("Px_先验=", Pxzz.T)
    Px_pos = Pxzz*Pxz[z,None].T
    Pxzz = Px_pos / sum(Px_pos)
    print("Px_后验=",Pxzz.T)