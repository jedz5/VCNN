import os
import numpy as np
import json
import pandas as pd
import traceback
import torch
from torch.utils.data import Dataset
from torch import nn
from collections import  Counter
import torch.nn.functional as F
import glob
"""
    [<class 'numpy.generic'>, [	
	[<class 'numpy.number'>, [
		[<class 'numpy.integer'>, [
			[<class 'numpy.signedinteger'>, [<class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, <class 'numpy.int64'>, <class 'numpy.int64'>, <class 'numpy.timedelta64'>]], 
			[<class 'numpy.unsignedinteger'>, [<class 'numpy.uint8'>, <class 'numpy.uint16'>, <class 'numpy.uint32'>, <class 'numpy.uint64'>, <class 'numpy.uint64'>]]]], 
		[<class 'numpy.inexact'>, [
			[<class 'numpy.floating'>, [<class 'numpy.float16'>, <class 'numpy.float32'>, <class 'numpy.float64'>, <class 'numpy.float128'>]], 
			[<class 'numpy.complexfloating'>, [<class 'numpy.complex64'>, <class 'numpy.complex128'>, <class 'numpy.complex256'>]]]]]], 
	[<class 'numpy.flexible'>, [
		[<class 'numpy.character'>, [<class 'numpy.bytes_'>, <class 'numpy.str_'>]], 
		[<class 'numpy.void'>, [<class 'numpy.record'>]]]], 
	<class 'numpy.bool_'>, 
	<class 'numpy.datetime64'>,
	<class 'numpy.object_'>]]
	
	torch.long() 将tensor投射为long类型

newtensor = tensor.long()
torch.half()将tensor投射为半精度浮点类型
newtensor = tensor.half()
torch.int()将该tensor投射为int类型
newtensor = tensor.int()
torch.double()将该tensor投射为double类型
newtensor = tensor.double()
torch.float()将该tensor投射为float类型
newtensor = tensor.float()
torch.char()将该tensor投射为char类型
newtensor = tensor.char()
torch.byte()将该tensor投射为byte类型
newtensor = tensor.byte()
torch.short()将该tensor投射为short类型
newtensor = tensor.short()
"""

class vcmi_Dataset(Dataset):
    def __init__(self,datapath):
        data = np.load(datapath)
        self.dataI = torch.tensor(data[0])
        self.dataV = torch.tensor(data[1]).float().unsqueeze(-1)
        self.y = torch.tensor(data[2])
        self.v = torch.tensor(data[3]).unsqueeze(-1)
        self.k_all = torch.tensor(data[4][:,:6].astype(np.int32))
        self.k_path = data[4][:,6]
    def __getitem__(self, idx):
        xI = self.dataI[idx]
        xv = self.dataV[idx]
        label_ka = self.y[idx,:,0]
        label_a = self.y[idx,:,1]
        label_k = self.y[idx, :, 2]
        label_v = self.v[idx]
        label_k_all = self.k_all[idx]
        k_path = self.k_path[idx]
        return xI,xv,label_ka,label_a,label_k,label_v,label_k_all,k_path
    def __len__(self):
        return len(self.dataI)

class fineTune_Dataset(Dataset):
    def __init__(self,datapath):
        self.files = sorted(glob.glob(datapath),reverse=True)
        self.path = datapath
    def __getitem__(self, idx):
        dataI = np.zeros((14,10))
        dataV = np.zeros((14, 10,1))
        label = np.zeros((7, 3))
        labelV = np.zeros((1))
        label_k_all = np.zeros([6])
        # try:
        fd = self.files[idx]
        with open(fd) as s:
            stackLocation = {}
            root = json.load(s)
            labelV[0] = root["win"]
            for st in root["stacks"]:
                slot = st["slot"] + (0 if st["isHuman"] else 7)
                dataI[slot, 0] = st["id"] + 1
                dataI[slot, 1] = st["speed"] - 2
                dataI[slot, 2] = st["luck"] + 4
                dataI[slot, 3] = st["morale"] + 4
                dataI[slot, 4] = 1
                dataI[slot, 5] = 1
                dataI[slot, 6] = 1
                dataI[slot, 7] = 1
                dataI[slot, 8] = 1
                dataI[slot, 9] = 1
                #
                dataV[slot, 0,0] = 1
                dataV[slot, 1,0] = 1
                dataV[slot, 2,0] = 1
                dataV[slot, 3,0] = 1
                dataV[slot, 4,0] = st["baseAmount"]
                dataV[slot, 5,0] = st["attack"]
                dataV[slot, 6,0] = st["maxDamage"]
                dataV[slot, 7,0] = st["minDamage"]
                dataV[slot, 8,0] = st["health"]
                dataV[slot, 9,0] = st["defense"]
                # 1 kill
                if st["isHuman"]:
                    label_k_all[0] += st["killed"]
                    label_k_all[1] += st["baseAmount"]
                    label_k_all[2] = st["id"]
                else:
                    label_k_all[3] += st["killed"]
                    label_k_all[4] += st["baseAmount"]
                    label_k_all[5] = st["id"]
                # 7 kills
                if st["isHuman"]:
                    if st["id"] not in stackLocation:
                        stackLocation[st["id"]] = st["slot"]
                    label[[stackLocation[st["id"]]], 0] += st["killed"]
                    label[[stackLocation[st["id"]]], 1] += st["baseAmount"]
            k_path = fd
        for slot in range(7):
            if label[slot, 1] == 0:
                label[slot, 2] = 0
                # label[slot, 3] = 0
            else:
                label[slot, 2] = label[slot, 0] / label[slot, 1]
                # label[slot, 3] = 1 / label[slot, 1]

        # except:
        #     traceback.print_exc()
        #     print("wrong file[{}] = {}".format(fd))
        #     # print("err {} rm file[{}] = {}".format(errCount,i,fd))
        #     s.close()
        #     return
            # os.remove(fd)
        dataI = torch.tensor(dataI,dtype=torch.int64)
        dataV = torch.tensor(dataV)
        label = torch.tensor(label)
        labelV = torch.tensor(labelV,dtype=torch.uint8)
        label_k_all = torch.tensor(label_k_all)
        return dataI,dataV,label[:,0],label[:,1],label[:,2],labelV,label_k_all,k_path
    def __len__(self):
        return len(self.files)
def storeTrainSimple(jsonsPath,outpath,NSamples = 6):
    dirs = os.listdir(jsonsPath)
    all_len = 0
    for dir in dirs:
        wd = os.path.join(jsonsPath,dir)
        files = os.listdir(wd)
        all_len += len(files)
        if all_len >= NSamples:
            break
    dataI = np.zeros([min(all_len, NSamples), 14,10 ], dtype=int)
    dataV = np.zeros([min(all_len, NSamples), 14,10 ], dtype=int)
    label = np.zeros([min(all_len, NSamples),7,3], dtype=float)
    labelV = np.zeros([min(all_len, NSamples)], dtype=int)
    label_k_all = np.zeros([min(all_len, NSamples),7], dtype=np.object)
    i = -1
    errCount = 0
    for dir in dirs:
        if i >= NSamples - 1:
            break
        wd = os.path.join(jsonsPath,dir)
        files = os.listdir(wd)
        for ii,a in enumerate(files):
            if i >= NSamples - 1:
                break
            i += 1
            try:
                fd = os.path.join(wd,a)
                with open(fd) as s:
                    stackLocation = {}
                    root = json.load(s)
                    labelV[i] = root["win"]
                    for st in root["stacks"]:
                        slot =  st["slot"] + (0 if st["isHuman"] else 7)
                        dataI[i, slot,0] = st["id"] + 1
                        dataI[i, slot , 1] = st["speed"] - 2
                        dataI[i, slot , 2] = st["luck"] + 4
                        dataI[i, slot , 3] = st["morale"] + 4
                        dataI[i, slot , 4] = 1
                        dataI[i, slot , 5] = 1
                        dataI[i, slot , 6] = 1
                        dataI[i, slot , 7] = 1
                        dataI[i, slot , 8] = 1
                        dataI[i, slot , 9] = 1
                        #
                        dataV[i, slot,0] = 1
                        dataV[i, slot , 1] = 1
                        dataV[i, slot , 2] = 1
                        dataV[i, slot , 3] = 1
                        dataV[i, slot , 4] = st["baseAmount"]
                        dataV[i, slot , 5] = st["attack"]
                        dataV[i, slot , 6] = st["maxDamage"]
                        dataV[i, slot , 7] = st["minDamage"]
                        dataV[i, slot , 8] = st["health"]
                        dataV[i, slot , 9] = st["defense"]
                        # 1 kill
                        if st["isHuman"]:
                            label_k_all[i,0] += st["killed"]
                            label_k_all[i, 1] += st["baseAmount"]
                            label_k_all[i, 2] = st["id"]
                        else:
                            label_k_all[i, 3] += st["killed"]
                            label_k_all[i, 4] += st["baseAmount"]
                            label_k_all[i, 5] = st["id"]
                        # 7 kills
                        if st["isHuman"]:
                            if st["id"] not in stackLocation:
                                stackLocation[st["id"]] = st["slot"]
                            label[i,[stackLocation[st["id"]]],0] += st["killed"]
                            label[i, [stackLocation[st["id"]]], 1] += st["baseAmount"]
                    label_k_all[i,6] = fd
                for slot in range(7):
                    if label[i, slot, 1] == 0:
                        label[i, slot, 2] = 0
                        # label[i, slot, 3] = 0
                    else:
                        label[i, slot, 2] = label[i, slot, 0]/label[i, slot, 1]
                        # label[i, slot, 3] = 1 / label[i, slot, 1]

            except:
                errCount += 1
                traceback.print_exc()
                s.close()
                # print("err {} file[{}] = {}".format(errCount, i, fd))
                print("err {} rm file[{}] = {}".format(errCount,i,fd))
                os.remove(fd)
                continue
    # print len([word for line in f for word in line.split()])
    np.save(outpath,[dataI,dataV,label,labelV,label_k_all,np.zeros([1,2])])
    # np.save(outpath,label_k_all)
    print("alllen = {}, {} samples completed erros {}".format(all_len,i+1,errCount))
    return label_k_all
def test(inXi,inXv):
    Xi = torch.tensor(inXi)
    Xv = torch.tensor(inXv).float().unsqueeze(-1)
    feature_sizes = [140, 24, 8, 8, 2, 2, 2, 2, 2, 2]
    embedding_size = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    fm_first_order_embeddings = nn.ModuleList(
        [nn.Embedding(feature_size, 1,padding_idx=0) for feature_size in feature_sizes])
    fm_second_order_embeddings = nn.ModuleList(
        [nn.Embedding(feature_size, embedding_size[idx],padding_idx=0) for idx, feature_size in enumerate(feature_sizes)])

    # fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for j in range(14) for i, emb in
    #                           enumerate(fm_first_order_embeddings)]

    # fm_first_order_emb_arr = [emb(Xi[:, i, j]) * Xv[:, i, j] for i in range(14) for j, emb in enumerate(fm_second_order_embeddings)]
    fm_first_order_emb_arr =[]
    for i in range(14):
        slot = []
        for j, emb in enumerate(fm_second_order_embeddings):
            slot.append(emb(Xi[:, i, j]) * Xv[:, i, j])
        fm_first_order_emb_arr.append(torch.cat(slot,1))

    slotEmb = nn.Linear(sum(embedding_size),16)
    slots = []
    for i in range(14):
        slot_i = slotEmb(fm_first_order_emb_arr[i])
        slot_i = F.relu(slot_i)
        slots.append(slot_i)
    slots = torch.cat(slots, 1)
    # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
    # fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
    #                            enumerate(fm_second_order_embeddings)]
    # fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
    print(slots)
    # print(fm_sum_second_order_emb)
if __name__ == "__main__":
    pass
    label_k_all = storeTrainSimple(r"/home/enigma/work/enigma/project/vcmi/RD/builds/samples_hand/te","../dataset/samples63_byhand_test.npy",2000000)
    # all = np.load("../dataset/samples63_byhand.npy")
    # label = all[4]
    # label2 = all[1]
    # print(label)







