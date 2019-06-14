import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter

import json
import pandas as pd
from DeepFM_model.model_vcmi import DeepFM_vcmi
from DeepFM_model.process_vcmi_data import vcmi_Dataset
import time
# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 300000

def run_train():
    batch = 1000
    train_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_train.npy')
    loader_train = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_test.npy')
    loader_val = DataLoader(val_data, batch_size=batch, shuffle=True)

    model = DeepFM_vcmi()
    resume = False
    start_epoch = 0
    best_start = 0.1
    if resume:
        checkpoint = torch.load('net_pram.pkl')
        model.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_start = checkpoint['best']
        print("init best = {} ".format(best_start))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    # start_epoch = 0
    # best_start = 0.13
    model.fit(loader_train, loader_val, optimizer, epochs=10, verbose=True, print_every=10, best_start=best_start,
              ep_start=start_epoch)
def initCreature():
    with open("../ENV/creatureData.json") as JsonFile:
        x = json.load(JsonFile)
        crList = [0] * (len(x["creatures"]) + 10)
        for item in x["creatures"]:
            crList[item["id"]]=item["name"]
        return pd.Series(crList)
def run_infer_inner(model,batch,path):
    crList = initCreature()
    val_data = vcmi_Dataset(path)
    loader_val = DataLoader(val_data, batch_size=batch)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        # dataiter = iter(loader_val)
        # xi, xv, y_ka, y_a, y_k,y_v,y_k_all = next(dataiter)
        results = [[],[],[]]
        for i,(xi, xv, y_ka, y_a, y_k,y_v,y_k_all) in enumerate(loader_val):

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
                df[("Enemy", "name")] = crList[y_k_all[idx_win, 5]].values
                df.iloc[:,[2,1,3,6,5,7]] = y_k_all[idx_win]
                return df
            since = time.time()
            pred_wrong = gendf()
            time_elapsed = time.time() - since
            print('init complete in {}s'.format(time_elapsed))
            results[0].append(y_k_all)
            results[1].append(pred_wrong)
        y_k_all_all = np.concatenate(results[0])
        pred_wrong = pd.concat(results[1])
        return y_k_all_all,pred_wrong
def get_counter(in_list):
    x = dict(Counter(in_list)) #
    xx = list((x.keys()))
    yy = list((x.values()))
    return xx,yy
def show_list(lists,colors):
    plt.figure()  # 定义一个图像窗口
    for l,c in zip(lists,colors):
        a = get_counter(l)
        plt.scatter(a[0], a[1], color=c)
    plt.show()
if __name__ == '__main__':
    model = DeepFM_vcmi(use_cuda=True)
    model = model.cuda()
    checkpoint = torch.load('net_pram.pkl')
    model.load_state_dict(checkpoint['net'])
    y_k_all, pred_wrong = run_infer_inner(model, 10000,'/home/enigma/work/enigma/project/VCNN/dataset/samples63_test.npy')
    y_filt = y_k_all[(y_k_all[:, 2] == 1) & (y_k_all[:, 5] == 11)]
    l = len(y_filt)
    since = time.time()
    lose_wrong = pred_wrong[pred_wrong.iloc[:, 1] > pred_wrong.iloc[:, 2]].sort_values(by=[("Me", "id"), ("Enemy", "id")])
    time_elapsed = time.time() - since
    print('sort complete in {}s'.format(time_elapsed))
    lose_filt = lose_wrong.loc[(lose_wrong.iloc[:, 3] == 1) & (lose_wrong.iloc[:, 7] == 11)]
    l2 = len(lose_filt)
    #
    s_y_k_all, s_pred_wrong = run_infer_inner(model, 100000,                                              '/home/enigma/work/enigma/project/VCNN/dataset/samples63_train.npy')
    s_y_filt = s_y_k_all[(s_y_k_all[:, 2] == 1) & (s_y_k_all[:, 5] == 11)]
    s_l = len(s_y_filt)
    since = time.time()
    s_lose_wrong = s_pred_wrong[s_pred_wrong.iloc[:, 1] > s_pred_wrong.iloc[:, 2]].sort_values(
        by=[("Me", "id"), ("Enemy", "id")])
    time_elapsed = time.time() - since
    print('sort complete in {}s'.format(time_elapsed))
    s_lose_filt = s_lose_wrong.loc[(s_lose_wrong.iloc[:, 3] == 1) & (s_lose_wrong.iloc[:, 7] == 11)]
    s_l2 = len(s_lose_filt)
    to_show = [s_y_filt[:, 1],s_lose_filt.iloc[:, 1],y_filt[:, 1],lose_filt.iloc[:, 1]] #
    colors = ["black","red","#5F9EA0","#00BFFF"] #
    show_list(to_show,colors)
    print(l2)