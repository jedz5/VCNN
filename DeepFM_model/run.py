import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# import matplotlib;matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter

from scipy.spatial.distance import cdist
import json
import pandas as pd
from DeepFM_model.model_vcmi import DeepFM_vcmi
from DeepFM_model.process_vcmi_data import vcmi_Dataset
import time
# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 300000

def run_train():
    batch = 2000
    train_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_train.npy')
    loader_train = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_data = vcmi_Dataset('/home/enigma/work/enigma/project/VCNN/dataset/samples63_test.npy')
    loader_val = DataLoader(val_data, batch_size=batch, shuffle=True)

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
    # start_epoch = 0
    best_start = 0.1
    return model.fit(loader_train, loader_val, optimizer, epochs=20, verbose=True, print_every=10, best_start=best_start,
              ep_start=start_epoch)
def initCreature(need_one_hot = False):
    with open("../ENV/creatureData.json") as JsonFile:
        x = json.load(JsonFile)
        crList = np.zeros((len(x["creatures"]) + 10),dtype=np.object)
        for item in x["creatures"]:
            crList[item["id"]]=item["name"]
        return pd.Series(crList)
def run_infer_inner(model,batch,path):
    crList = initCreature()
    val_data = vcmi_Dataset(path)
    loader_val = DataLoader(val_data, batch_size=batch,shuffle=True)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        # dataiter = iter(loader_val)
        # xi, xv, y_ka, y_a, y_k,y_v,y_k_all = next(dataiter)
        results = [[],[],[]]
        for i,(xi, xv, y_ka, y_a, y_k,y_v,y_k_all,k_path) in enumerate(loader_val):

            xi = xi.to(device=model.device, dtype=model.dtype)  # move to device, e.g. GPU
            xv = xv.to(device=model.device, dtype=torch.float)
            y_ka = y_ka.to(device=model.device, dtype=torch.float)
            y_a = y_a.to(device=model.device, dtype=torch.float)
            mask = (y_a != 0).to(device=model.device, dtype=torch.float)
            y_k_all = y_k_all.numpy()
            win,preds = model(xi, xv)
            preds_clip = torch.clamp(preds,0,1.0) * mask
            pred = torch.round(preds_clip * y_a)
            total = torch.abs(pred - y_ka) / (y_a + 1E-6)
            win = win.cpu().numpy()
            y_v = y_v.numpy()
            xi = xi.cpu().numpy()
            idx_win = np.where((win > 0.5) != y_v)[0]
            neg_sample_len = len(idx_win)
            def analy_2():
                # idx_sort = np.argsort(-total.cpu().numpy().sum(axis=1))
                total_sort = total.cpu().numpy().sum(axis=1)
                pred_sort = pred.cpu().numpy().sum(axis=1)
                y_ka_sort = y_ka.cpu().numpy().sum(axis=1)
                win_sort = win.squeeze()
                y_v_sort = y_v.squeeze()
                y_k_all_sort = y_k_all.astype(np.object)
                names_sort = [crList[y_k_all_sort[:, 2]],crList[y_k_all_sort[:,5]]]
                names_sort = np.array(names_sort, dtype=object).transpose()
                # y_k_all_sort_insert = np.insert(y_k_all_sort,[2,5],names_sort,axis=1)
                y_k_all_sort[:,[2,5]] = names_sort
                total_sort = y_v_sort * total_sort
                pred_wrong = (win_sort > 0.5) != y_v_sort
                total_sort[pred_wrong] = 1
                y_k_all_s = np.insert(y_k_all_sort,[0],np.around(np.array([win_sort,total_sort,pred_sort],dtype=np.float).transpose(),decimals=3),axis=1)
                y_k_all_pd = pd.DataFrame(y_k_all_s,columns=["预测胜率","错误率","预测伤亡","我方伤亡","我方总数","我方","敌方伤亡","敌方总数","敌方"])
                y_k_all_sort_pd = y_k_all_pd.sort_values(by=["错误率","我方"],ascending=False)
                cc = Counter(y_k_all[pred_wrong, 2])
                cr_id = list(cc.keys())
                cr_num = list(cc.values())
                statis = pd.DataFrame({'生物':crList[cr_id],'错次':cr_num}).sort_values(by='错次')
                print() #y_k_all_sort[((win_sort > 0.5) != y_v_sort),1] = 1
            analy_2()
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
            pred_wrong_list = gendf()
            time_elapsed = time.time() - since
            print('init complete in {}s'.format(time_elapsed))
            results[0].append(y_k_all)
            results[1].append(pred_wrong_list)
        y_k_all_all = np.concatenate(results[0])
        pred_wrong_list = pd.concat(results[1])
        return y_k_all_all,pred_wrong_list
def get_counter(in_list):
    x = dict(Counter(in_list)) #
    xx = list((x.keys()))
    yy = list((x.values()))
    return xx,yy
def show_list(lists,markers,type = None):
    plt.figure(figsize=(16, 10))  # 定义一个图像窗口
    if type == "set":
        for l,m in zip(lists,markers):
            s = list(set(l))
            y = [1] * len(s)
            plt.scatter(s, y,marker= m,markersize=1)
    elif type == "points":
        for l,m in zip(lists,markers):
            plt.scatter(l[0],l[1],marker= m)
    else:
        for l,m in zip(lists,markers):
            a = get_counter(l)
            plt.scatter(a[0], a[1],marker= m)
    plt.show()
def run_infer():
    model = DeepFM_vcmi(use_cuda=False)
    # model = model.cuda()
    checkpoint = torch.load('net_pram.pkl')
    model.load_state_dict(checkpoint['net'])
    y_k_all, pred_wrong = run_infer_inner(model, 50000,
                                          '/home/enigma/work/enigma/project/VCNN/dataset/samples63_valid.npy')
    y_filt = y_k_all[(y_k_all[:, 2] == 1) & (y_k_all[:, 5] == 11)]
    l = len(y_filt)
    since = time.time()
    lose_wrong = pred_wrong[pred_wrong.iloc[:, 1] > pred_wrong.iloc[:, 2]].sort_values(
        by=[("Me", "id"), ("Enemy", "id")])
    time_elapsed = time.time() - since
    print('sort complete in {}s'.format(time_elapsed))
    lose_filt = lose_wrong.loc[(lose_wrong.iloc[:, 3] == 1) & (lose_wrong.iloc[:, 7] == 11)]
    l2 = len(lose_filt)
    #
    s_y_k_all, s_pred_wrong = run_infer_inner(model, 100000,
                                              '/home/enigma/work/enigma/project/VCNN/dataset/samples63_train.npy')
    s_y_filt = s_y_k_all[(s_y_k_all[:, 2] == 1) & (s_y_k_all[:, 5] == 11)]
    s_l = len(s_y_filt)
    since = time.time()
    s_lose_wrong = s_pred_wrong[s_pred_wrong.iloc[:, 1] > s_pred_wrong.iloc[:, 2]].sort_values(
        by=[("Me", "id"), ("Enemy", "id")])
    time_elapsed = time.time() - since
    print('sort complete in {}s'.format(time_elapsed))
    s_lose_filt = s_lose_wrong.loc[(s_lose_wrong.iloc[:, 3] == 1) & (s_lose_wrong.iloc[:, 7] == 11)]
    s_l2 = len(s_lose_filt)
    h5 = pd.HDFStore("../dataset/pred.h5",'w')
    h5["pred_valid"] = pred_wrong
    h5["pred_source"] = s_pred_wrong
    print(len(pred_wrong))
    print(len(s_pred_wrong))
    h5.close()
def analy():
    y = np.load("../dataset/samples63_test_only_y.npy")
    sy = np.load("../dataset/samples63_train_only_y.npy")
    h5 = pd.HDFStore("../dataset/pred.h5", 'r')
    pred_wrong = h5["pred_valid"]
    s_pred_wrong = h5["pred_source"]
    Me = 121
    Enemy = 119
    b = y[(y[:, 2] == Me) & (y[:, 5] == Enemy)]
    sb = sy[(sy[:, 2] == Me) & (sy[:, 5] == Enemy)]
    a = pred_wrong.loc[(pred_wrong.iloc[:, 3] == Me) & (pred_wrong.iloc[:, 7] == Enemy)]

    show_list([(sb[:, 1], sb[:, 4]), (b[:, 1], b[:, 4]), (a.iloc[:, 1].values, a.iloc[:, 5].values)],
              markers=['x', '_', 'o'], type="points")
    h5.close()


def to_embeds():
    NumUpdateCreatures = 63
    creIDs = np.zeros(NumUpdateCreatures)
    # cre_embs = np.zeros((NumUpdateCreatures,10),dtype=np.float32)
    for i in range(NumUpdateCreatures):
        creIDs[i] = i * 2 + 1;
    creIDs[56] = 119;
    creIDs[57] = 127;
    creIDs[58] = 123;
    creIDs[59] = 129;
    creIDs[60] = 125;
    creIDs[61] = 121;
    creIDs[62] = 131;
    creIDs += 1
    cr_names = initCreature()
    model = DeepFM_vcmi(use_cuda=False)
    # model = model.cuda()
    checkpoint = torch.load('net_pram.pkl')
    model.load_state_dict(checkpoint['net'])
    cre_embs = model.creature_prop_embeddings[0](torch.tensor(creIDs).long()).detach().numpy()
    cre_embs = pd.DataFrame(cre_embs,index=creIDs.tolist())
    # corr = cre_embs.corr()
    # x = pd.concat([corr[14], cr_names], axis=1)
    # x = x.sort_values(by=14)
    dis = cdist(cre_embs, cre_embs, metric='euclidean')
    # creIDs2 = creIDs - 1
    # cre_embs2 = model.creature_prop_embeddings[0](torch.tensor(creIDs2).long()).detach().numpy()
    # cre_embs2 = pd.DataFrame(cre_embs2, index=creIDs2.tolist()).transpose()
    # corr2 = cre_embs2.corr()
    # cr_names2 = cr_names.drop(0).reset_index(drop=True)
    # x2 = pd.concat([corr2[13], cr_names2], axis=1)
    # x2 = x2.sort_values(by=13)
    a = cr_names[creIDs[np.argsort(-dis[6])]]
    print(a)
if __name__ == '__main__':
    state = run_infer()
