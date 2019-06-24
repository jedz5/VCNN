import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
import timeit
import pdb

import matplotlib.pyplot as plt
from collections import  Counter
# from DeepFM_model.run import show_list
x = np.random.gamma(2,2,1000) + 4
x = x.astype(np.int32)
x = dict(Counter(x))
xx = list((x.keys()))
yy = list((x.values()))
plt.figure()    # 定义一个图像窗口
plt.scatter(xx,yy) # 绘制曲线 y1
plt.show()