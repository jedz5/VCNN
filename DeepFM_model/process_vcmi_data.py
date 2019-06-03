import os
import numpy as np
import json
import pandas as pd
def storeTrainSimple(jsonsPath):
    fils = os.listdir(jsonsPath)
    NSamples = 600
    data = np.zeros([min(len(fils), NSamples), 10 * 14], dtype=int)
    label = np.zeros([min(len(fils), NSamples), 7], dtype=int)
    others = pd.DataFrame(np.zeros(min(len(fils), NSamples),8),index=[[]])#[[[0,0],[0,0],[0,0],[0,0]] for x in range(min(len(fils), NSamples))]
    for i,a in enumerate(fils):
        if i >= NSamples:
            break
        with open(jsonsPath+a) as s:
            stackLocation = {}
            root = json.load(s)
            oa = False
            ob = False
            for st in root["stacks"]:
                slot = 10 * (st["slot"] + (0 if st["isHuman"] else 7))
                data[i, slot] = st["id"] + 1
                data[i, slot + 1] = st["baseAmount"]
                data[i, slot + 2] = st["speed"] - 2
                data[i, slot + 3] = st["luck"] + 4
                data[i, slot + 4] = st["morale"] + 4
                data[i, slot + 5] = st["attack"]
                data[i, slot + 6] = st["maxDamage"]
                data[i, slot + 7] = st["minDamage"]
                data[i, slot + 8] = st["health"]
                data[i, slot + 9] = st["defense"]
                # if st["isHuman"]:
                #     if st["id"] not in stackLocation:
                #         stackLocation[st["id"]] = st["slot"]
                #     label[i,[stackLocation[st["id"]]]] += st["killed"]
                if st["isHuman"]:
                    others[i][0][0] = st["name"]
                    others[i][1][0] += st["baseAmount"]
                    others[i][2][0] += st["killed"]
                    others[i][3][0] = st["id"]
                else:
                    others[i][0][1] = st["name"]
                    others[i][1][1] += st["baseAmount"]
                    others[i][2][1] += st["killed"]
                    others[i][3][1] = st["id"]
            # others[i][2][0] = sum(label[i][:7])
            # others[i][2][1] = sum(label[i][7:])
    def isDead(z):
        for y in z:
            if y > 56 and y<72:
                return 0
        return 1
    # f = open('test.txt', 'r')
    # print len([word for line in f for word in line.split()])
    # z = [isDead(x) for x in data[list(set(np.where(data[:, 4::10] == 0)[0].tolist()))][:, 0::10]]
    # print(sum(z))
    print(others)
    # np.save("../dataset/samples63.npy",data,label)
if __name__ == "__main__":
    data = storeTrainSimple("/home/enigma/work/enigma/project/vcmi/RD/install/train0/")











# import torch
# from torch.autograd import Variable
# from torch.utils.data import dataloader
# from torch.utils.data import dataset
# import os
# import json
# import numpy as np
# class VCMIData(dataset):
#     def __init__(self,path):
#         self.datapath = path
#         self.jsons = os.scandir(path)
#     def __getitem__(self, idx):
#         with open(self.jsons[idx]) as a:
#             root = json.load(a)
#             for st in root["stacks"]:
#                 slot = 10 * (st["slot"] + 0 if st["isHuman"] else 7)
#                 data[i, slot] = st["id"]
#                 data[i, slot + 1] = st["baseAmount"]
#                 data[i, slot + 2] = st["attack"]
#                 data[i, slot + 3] = st["luck"]
#                 data[i, slot + 4] = st["morale"]
#                 data[i, slot + 5] = st["speed"]
#                 data[i, slot + 6] = st["maxDamage"]
#                 data[i, slot + 7] = st["minDamage"]
#                 data[i, slot + 8] = st["health"]
#                 data[i, slot + 9] = st["defense"]
#                 if st["isHuman"]:
#                     if st["id"] not in stackLocation:
#                         stackLocation[st["id"]] = st["slot"]
#                     label[i, [stackLocation[st["id"]]]] += st["baseAmount"]