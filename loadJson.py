import json
import os
import traceback
import numpy as np
import CNNNV as cnn
hexY = 11
hexX = 17
hexDepth = 16
def load(inFile):
    root = None
    with open(inFile) as jsonFile:
        root = json.load(jsonFile)
        plane = np.zeros((hexY,hexX,hexDepth))
        label = np.zeros((8))
       # for x in range(hexX):
           # for y in range(hexY):
        try:
            if 'obstacles' in root:
                for x in root['obstacles']:
                    plane[x['y']][x['x']][0] = 0
                    plane[x['y']][x['x']][1] = 1
            for x in root['stacks']:
                if x['isHuman']:
                    plane[x['y']][x['x']][0] = 1
                    plane[x['y']][x['x']][1] = 0
                    if x['isWide']:
                        plane[x['y']][x['x'] - 1][0] = 1
                        plane[x['y']][x['x'] - 1][1] = 0
                else:
                    plane[x['y']][x['x']][0] = -1
                    plane[x['y']][x['x']][1] = 0
                plane[x['y']][x['x']][2] = x['amount']
                plane[x['y']][x['x']][3] = x['attack']
                plane[x['y']][x['x']][4] = x['defense']
                plane[x['y']][x['x']][5] = x['maxDamage']
                plane[x['y']][x['x']][6] = x['minDamage']
                plane[x['y']][x['x']][7] = x['firstHPleft']
                plane[x['y']][x['x']][8] = x['health']
                plane[x['y']][x['x']][9] = x['speed']
                plane[x['y']][x['x']][10] = x['isCanShoot']
                plane[x['y']][x['x']][11] = x['isCanMove']
                plane[x['y']][x['x']][12] = x['isMoved']
                plane[x['y']][x['x']][13] = x['isRetaliate']
                plane[x['y']][x['x']][14] = x['isWaited']
                plane[x['y']][x['x']][15] = x['isWide']
        except:
            traceback.print_exc()
        label[root['action']['actionType'] - 1] = 1.0
    return plane,label

def loadData(path):
    batch = []
    y = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == ".json":
            batch.append(load(i)[0])
            y.append(load(i)[1])
    print("end")
    bx = np.asarray(batch)
    by = np.asarray(y)
    return bx,by
if __name__ == "__main__":
    batch = []
    y = []
    f_list = os.listdir(".")
    # np.random.shuffle(f_list)
    # np.save("./result/shuffle",f_list)
    #f1_list =np.load("./result/shuffle.npy")
    for i in f_list:
        if os.path.splitext(i)[1] == ".json":
            batch.append(load(i)[0])
            y.append(load(i)[1])
    bx = np.asarray(batch)
    by = np.asarray(y)
    cnn.train(bx,by)
# matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)
    # sess = tf.Session()
    # result = sess.run(product)
    # print result
    # sess.close()
