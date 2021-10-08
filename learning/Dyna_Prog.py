# 01背包问题
import numpy as np
W = 20
w = [10,1,1,1,5,1,2,8,4,2]
v = [20,1,2,3,8,2,6,10,7,3]
dp = np.zeros((len(w)+1,W+1))
for i in range(len(w)):
    dp[i + 1, :] = dp[i,:]
    for j in range(w[i],W):
        tmp = dp[i,j - w[i]] + v[i]
        dp[i+1,j] = max(dp[i,j],tmp)
        print(dp)



# 01背包问题(空间优化版)
import numpy as np
W = 20
w = [10,1,1,1,5,2,8,4,2]
v = [20,1,2,3,8,6,10,7,3]
dp = np.zeros((W+1,))
for i in range(len(w)):
    for j in range(W,w[i]-1,-1): # 必须逆向枚举!!!
        tmp = dp[j - w[i]] + v[i]
        dp[j] = max(dp[j],tmp)
        print(dp)

