import numpy as np
from collections import Counter
import json
import traceback
import os
import glob
def simple_run():
    train_data = simple_als(r'D:\project\vcmi\RD\train\*.json')
    killed_2 = np.array(train_data[:, 0, 1:, 1])
    killed_1 = np.array(train_data[:, 0, 0, 1])
    amount_1 = np.array(train_data[:, 0, 0, 0])
    killed_2 = killed_2.sum(axis=-1)
    amount_2 = np.array(train_data[:, 0, 1:, 0])
    amount_2 = amount_2.sum(axis=-1)
    smy = np.stack([amount_1,killed_1],axis=1)
    smy2 = np.stack([amount_2, killed_2], axis=1)
    killed_2 = np.array(train_data[:, 1, :, 1])
    killed_2 = killed_2.sum(axis=-1)
    amount_2 = np.array(train_data[:, 1, :, 0])
    amount_2 = amount_2.sum(axis=-1)
    smy3 = np.stack([amount_2, killed_2],axis=1)
    smy_0 = np.stack([smy,smy2,smy3],axis=1).reshape(-1,6)
    print()
def simple_als(path):
    ix=0
    fds = glob.glob(path)
    plane = np.zeros((len(fds), 2, 7, 2))
    for inFile in fds:
        with open(inFile) as jsonFile:
            root = json.load(jsonFile)
            try:
                if root['quickBattle']:
                    return
                for x in root['stacks']:
                    plane[ix,~x['isHuman'],x['slot'],0] = x['baseAmount']
                    plane[ix,~x['isHuman'],x['slot'],1] = x['killed']

            except:
                traceback.print_exc()
                return
        ix += 1
    return plane
if __name__ == '__main__':
    simple_run()