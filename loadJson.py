import json
import os
import traceback
import numpy as np
import CNNNV as cnn
import tensorflow as tf
import sys
print(sys.path)
side = 2
stacks = 7
hexDepth = 7
def load(inFile):
    root = None
    with open(inFile) as jsonFile:
        root = json.load(jsonFile)
        plane = np.zeros((side,stacks,hexDepth))
        label = np.zeros((9,1))
       # for x in range(hexX):
           # for y in range(hexY):
        try:
            for x in root['stacks']:
                plane[x['id0']][x['slot']][0] = x['id2']
                plane[x['id0']][x['slot']][1] = x['id3']

                plane[x['id0']][x['slot']][1] = (x['id4'] == 15)
                plane[x['id0']][x['slot']][1] = (x['id5'] == 25)
                plane[x['id0']][x['slot']][1] = (x['id6'] == 35)
                plane[x['id0']][x['slot']][1] = (x['id7'] == 45)

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

def quick(jsonData):
    print(jsonData)
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()
    return jsonData
if __name__ == "__main__":
    # batch = []
    # y = []
    # f_list = os.listdir(".")
    # # np.random.shuffle(f_list)
    # # np.save("./result/shuffle",f_list)
    # #f1_list =np.load("./result/shuffle.npy")
    # for i in f_list:
    #     if os.path.splitext(i)[1] == ".json":
    #         batch.append(load(i)[0])
    #         y.append(load(i)[1])
    # bx = np.asarray(batch)
    # by = np.asarray(y)
    # cnn.train(bx,by)
    quick("hello")


# matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)
    # sess = tf.Session()
    # result = sess.run(product)
    # print result
    # sess.close()
