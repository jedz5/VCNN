import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import json
import pandas as pd
from DeepFM_model.model_vcmi import DeepFM_vcmi
from DeepFM_model.process_vcmi_data import vcmi_Dataset
import timeit
import time
import pdb
# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 300000

def run_train():
    batch = 10000
    train_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_train.npy')
    loader_train = DataLoader(train_data, batch_size=batch, sampler=sampler.SubsetRandomSampler(range(Num_train)))
    val_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_test.npy')
    loader_val = DataLoader(val_data, batch_size=batch, sampler=sampler.SubsetRandomSampler(range(30000)))

    model = DeepFM_vcmi()
    resume = True
    start_epoch = 0
    best_start = 0.1
    if resume:
        checkpoint = torch.load('net_pram.pkl')
        model.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_start = checkpoint['best']
        print("init best = {} ".format(best_start))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    start_epoch = 0
    # best_start = 0.13
    model.fit(loader_train, loader_val, optimizer, epochs=10, verbose=True, print_every=1, best_start=best_start,
              ep_start=start_epoch)
def initCreature():
    with open("../ENV/creatureData.json") as JsonFile:
        x = json.load(JsonFile)
        crList = [0] * (len(x["creatures"]) + 10)
        for item in x["creatures"]:
            crList[item["id"]]=item["name"]
        return pd.Series(crList)
def run_infer():
    crList = initCreature()
    val_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_test.npy')
    loader_val = DataLoader(val_data, batch_size=30000, sampler=sampler.SubsetRandomSampler(range(30000)))
    model = DeepFM_vcmi(use_cuda=False)
    # model = model.cuda()
    checkpoint = torch.load('net_pram.pkl')
    model.load_state_dict(checkpoint['net'])
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        dataiter = iter(loader_val)
        xi, xv, y_ka, y_a, y_k,y_v,y_k_all = next(dataiter)
        xi = xi.to(device=model.device, dtype=model.dtype)  # move to device, e.g. GPU
        xv = xv.to(device=model.device, dtype=torch.float)
        y_ka = y_ka.to(device=model.device, dtype=torch.float)
        y_a = y_a.to(device=model.device, dtype=torch.float)
        mask = (y_a != 0).to(device=model.device, dtype=torch.float)
        y_k_all = y_k_all.numpy()
        win,preds = model(xi, xv)
        preds = preds * mask
        pred = torch.round(preds * y_a)
        total = torch.abs(pred - y_ka) / (y_a + 1E-6)
        win = win.cpu().numpy()
        y_v = y_v.numpy()
        xi = xi.cpu().numpy()
        idx_win = np.where((win > 0.5) != y_v)[0]
        neg_sample_len = len(idx_win)
        def gendf():
            cols = pd.MultiIndex.from_product([["Me", "Enemy"],["name","baseAmount", "killed", "id"]],names=["side","props"])
            df = pd.DataFrame(np.ones([neg_sample_len,4 * 2],dtype=np.uint16),
                              columns=cols)
            df[("Me", "name")] = df[("Me", "name")].astype(np.object)
            df[("Enemy", "name")] = df[("Me", "name")].astype(np.object)
            df[("Me", "id")] = df[("Me", "id")].astype(np.uint8)
            df[("Enemy", "id")] = df[("Me", "id")].astype(np.uint8)

            df[("Me","name")] = crList[y_k_all[idx_win,2]].values
            # df[("Me", "killed")] = y_k_all[idx_win, 0]
            # df[("Me", "baseAmount")] = y_k_all[idx_win,1]
            # df[("Me", "id")] = y_k_all[idx_win,2]
            df[("Enemy", "name")] = crList[y_k_all[idx_win, 5]].values
            # df[("Enemy", "killed")] = y_k_all[idx_win, 3]
            # df[("Enemy", "baseAmount")] = y_k_all[idx_win, 4]
            # df[("Enemy", "id")] = y_k_all[idx_win, 5]
            # df.iloc[:,[0,4]] = crList[y_k_all[idx_win, [2,5]]].values
            df.iloc[:,[2,1,3,6,5,7]] = y_k_all[idx_win]
            return df
        since = time.time()
        df = gendf()
        time_elapsed = time.time() - since
        print('init complete in {}s'.format(time_elapsed))
        since = time.time()
        lose_wrong = df[df.iloc[:, 1] > df.iloc[:, 2]].sort_values(by=[("Me", "id"), ("Enemy", "id")])
        time_elapsed = time.time() - since
        print('sort complete in {}s'.format(time_elapsed))
        print(lose_wrong.head())
if __name__ == '__main__':
    run_infer()
    # cl = initCreature()
    # print(cl)