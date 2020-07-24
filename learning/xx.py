import torch
import time
import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.data import Batch
from pstats import SortKey


# import pstats
# pcpu=pstats.Stats(r"D:\doc\pstats\cpu4w.pstat")
# pcuda=pstats.Stats(r"D:\doc\pstats\local_cpu.pstat")
# pcpu.sort_stats("cumulative")
# pcuda.sort_stats("cumulative")
# pcpu.print_stats("h3_ppo","forward")  #("h3_ppo","__call__")
# pcuda.print_stats("h3_ppo","forward")  #("h3_ppo","__call__")
#
# pcpu.print_callees("h3_ppo","forward")  #("h3_ppo","__call__")  # 可以显示哪个函数调用了哪些函数
# pcuda.print_callees("h3_ppo","forward")  #("h3_ppo","__call__")


a = torch.tensor([[1,2],[3,4]])
b = torch.argmax(a,dim=-1)[0].item()