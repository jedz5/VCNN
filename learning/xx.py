import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
import timeit
import pdb
import matplotlib.pyplot as plt
from collections import  Counter
x = [1,2,3,4,1]
x = dict(Counter(x))
xx = list((x.keys()))
yy = list((x.values()))
plt.figure()    # 定义一个图像窗口
plt.scatter(xx,yy) # 绘制曲线 y1
# plt.show()
x = [3,4,5,6,3]
x = dict(Counter(x))
xx = list((x.keys()))
yy = list((x.values()))
plt.scatter(xx,yy)
# plt.yticks(list(range(1,5)))
plt.show()