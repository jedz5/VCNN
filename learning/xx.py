import pandas as pd
import numpy as np


df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':["asdf", "ddf", "asaaaadf", "asdggf"],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])
N = 10
b = np.ones([N,3])

df3 = pd.DataFrame(b,index=[[],["Me","Enemy"]*int(N/2)],columns=["baseAmount","killed","id"])
# df3.insert(0,"name",a)
print(df3)