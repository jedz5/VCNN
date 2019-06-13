import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
import timeit
import pdb
# a = np.random.rand(10000)
# b = np.random.rand(10000)
# rec = np.rec.fromarrays([a,b])
# def test():
#       rec.argsort(order='f0')
# def test2():
#       rec['f0'].argsort()
# rest = timeit.repeat(stmt=test2,number=100,repeat=10,)
# print(min(rest))
# print(max(rest))
# print(rest)

# df = pd.DataFrame([['1.1', 2, 3]] * 2, dtype='O')
# print(df.dtypes)
# print(df)
# df.apply(pd.to_numeric)
# print(df.dtypes)
# print(df)
def test3():
      cols = pd.MultiIndex.from_product([["Me", "Enemy"],["name","baseAmount", "killed", "id"]],names=["side","props"])
      df = pd.DataFrame(np.zeros([300000,4 * 2],dtype=np.uint16),columns=cols)
      # df.iloc[:,[0,4]] = df.iloc[:,[0,4]].astype(np.object)
      # df.iloc[:,[3,7]] = df.iloc[:,[3,7]].astype(np.uint8)
      df[("Me", "name")] = df[("Me", "name")].astype(np.object)
      df[("Enemy","name")] = df[("Me","name")].astype(np.object)
      df[("Me","id")] = df[("Me","id")].astype(np.uint8)
      df[("Enemy","id")] = df[("Me","id")].astype(np.uint8)
      return df
      # print(df.dtypes)
N = 50
rest = timeit.timeit(stmt=test3,number=N)
print(rest/N)
df = test3()
pdb.set_trace()
# test3()