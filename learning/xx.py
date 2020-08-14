import pandas as pd

df=pd.read_excel('d:/土地款支付统计.xlsx',sheet_name='Sheet2')
#数据 偶数行 4-60列
a = df.iloc[::2, 4:60]
result = {}
#2014-2019
for i in range(2014,2020):
    # 日期 奇数行
    a1 = df.iloc[1::2,4:60].copy().to_numpy()
    b = ((pd.Timestamp(f'{i-1}-12-14 00:00:00') < a) & (a < pd.Timestamp(f'{i}-12-15 00:00:00'))).to_numpy()
    a1[~b] = 0
    result[i] = a1.sum(axis=-1)

#2020 1-6月
a1 = df.iloc[1::2,4:60].copy().to_numpy()
b = ((pd.Timestamp(f'2019-12-14 00:00:00') < a) & (a < pd.Timestamp(f'2020-6-15 00:00:00'))).to_numpy()
a1[~b] = 0
result["1-6"] = a1.sum(axis=-1)

#2020 7-12月
for i in range(7,13):
    a1 = df.iloc[1::2,4:60].copy().to_numpy()
    b = ((pd.Timestamp(f'2020-{i-1}-14 00:00:00') < a) & (a < pd.Timestamp(f'2020-{i}-15 00:00:00'))).to_numpy()
    a1[~b] = 0
    result[i] = a1.sum(axis=-1)
# 写入新文件
df2 = df.copy()
ii = 0
for k in result.keys():
    #插入对应列
    df2.insert(3 + ii,k,0)
    df2.iloc[1::2,3 + ii] = result[k]
    ii += 1
df2.to_excel('d:/hhh.xlsx')