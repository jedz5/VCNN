import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F

# num_FC = 200
# a = torch.ones(64,2048,device='cpu')
# b = torch.ones(num_FC,2048,1024,device='cpu')
# st = time.time()
# c = torch.matmul(a, b)
# print("cpu compute cost={}".format(time.time() - st))
# # a.cuda()
#
# a = torch.randn(64,2048,device='cpu')
# b = torch.randn(num_FC,2048,1024,device='cpu')
# st = time.time()
# a = a.cuda()
# b = b.cuda()
# st2 = time.time()
# c = torch.matmul(a, b)
# print("cuda transfor cost={}, compute cost={}".format(st2 -st,time.time() - st))
# #
# #
# l = []
# a = torch.randn(64,2048,device='cpu')
# b = [torch.randn(2048,1024,device='cpu') for i in range(num_FC)]
# str = ""
# st = time.time()
# a = a.cuda()
# # str += " trs {}".format(time.time() - st)
# for i in b:
#     st0 = time.time()
#     i = i.cuda()
#     ed0 = time.time()
#     # str += " trs {}".format(ed0 - st0)
#     c = torch.matmul(a, i)
#     # str += " cpt {}".format(time.time() - ed0)
#     l.append(c)
# # print(str)
# print("cuda compute cost={}".format(time.time() - st))

# l = []
# a = torch.ones(64,2048,device='cpu')
# b = [torch.ones(2048,1024,device='cpu') for i in range(11)]
# st = time.time()
# a = a.to(device='cuda')
# for i in b:
#     i = i.to(device='cuda')
#     c = torch.matmul(a, i)
#     l.append(c)
# print(time.time() - st)
#
#
# l = []
# a = torch.ones(64,2048,device='cpu')
# b = [torch.ones(2048,1024,device='cpu') for i in range(num_FC)]
# st = time.time()
# c = [torch.matmul(a, i) for i in b]
# print(time.time() - st)




# import torch.sparse
# nnz = 3
# dims = [5, 5, 2, 3]
# mask_targets = [2,3]
# value_size = len(mask_targets)
# I = torch.tensor([[0] * value_size,mask_targets])
# mask_targets_1 = torch.sparse_coo_tensor(I,torch.ones(value_size),(1,7))
#
# mask_targets = [1,4,6]
# value_size = len(mask_targets)
# I = torch.tensor([[0] * value_size,mask_targets])
# mask_targets_2 = torch.sparse_coo_tensor(I,torch.ones(value_size),(1,7))
# m = torch.cat([mask_targets_1,mask_targets_2],0)

a = np.ones([2,3])
b = (0 < a) & (a < 20)
print(b)