import torch
from torch.autograd import Variable
from torch.utils.data import dataloader
from torch.utils.data import dataset
import os
import json
import numpy as np

# emb = torch.nn.Embedding(56,4,padding_idx=55)
# data = np.array([[55,2,3,4],[5,6,7,8]])
# ten = data[:,::2]
# print(ten)
# input = Variable(torch.LongTensor(ten))
# print(emb(input))

class VCMIData(dataset):
    def __init__(self,path):
        self.datapath = path
        self.jsons = os.scandir(path)
    def __getitem__(self, idx):
        with open(self.jsons[idx]) as a:
            root = json.load(a)
            for st in root["stacks"]:
                slot = 10 * (st["slot"] + 0 if st["isHuman"] else 7)
                data[i, slot] = st["id"]
                data[i, slot + 1] = st["baseAmount"]
                data[i, slot + 2] = st["attack"]
                data[i, slot + 3] = st["luck"]
                data[i, slot + 4] = st["morale"]
                data[i, slot + 5] = st["speed"]
                data[i, slot + 6] = st["maxDamage"]
                data[i, slot + 7] = st["minDamage"]
                data[i, slot + 8] = st["health"]
                data[i, slot + 9] = st["defense"]
                if st["isHuman"]:
                    if st["id"] not in stackLocation:
                        stackLocation[st["id"]] = st["slot"]
                    label[i, [stackLocation[st["id"]]]] += st["baseAmount"]

