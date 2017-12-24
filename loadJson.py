import json
import os
import traceback
import numpy as np
import sys
print(sys.path)
side = 2
stacks = 7
hexDepth = 3
n_classes = 8
n_spells = 4
n_manaCost = 1
n_all = side*stacks*hexDepth+n_spells+n_manaCost
def load(inFile):
    root = None
    with open(inFile) as jsonFile:
        root = json.load(jsonFile)
        plane = np.zeros((side,stacks,hexDepth))
        label = np.zeros((n_classes,1))
        amount = np.zeros((n_classes,1))
        value = np.zeros((n_classes,1))
        spells = {'26':1,'41':2,'53':3,'54':4}
        try:
            if not 'hero' in root:
                return
            heroStrength = np.math.sqrt((1 + 0.05 * root['hero']['attack']) * (1 + 0.05 * root['hero']['defense']))
            for x in root['stacks']:
                plane[~x['isHuman']][x['slot']][0] = x['baseAmount']
                plane[~x['isHuman']][x['slot']][1] = x['fightValue']*heroStrength
                plane[~x['isHuman']][x['slot']][2] = x['aiValue']*heroStrength
                if x['isHuman']:
                    label[x['slot']] = x['killed']
                    amount[x['slot']] = x['baseAmount']
                    value[x['slot']] = x['aiValue']
            plane.resize((n_all,),refcheck=False)
            for y in root['hero']['spells']:
                if str(y['id']) in spells:
                    plane[spells[str(y['id'])]+n_all-1] = 1

        except:
            traceback.print_exc()
            return
        label[7][0] = root['manaCost']
        amount_[7][0] = root['manaCost']
        #label[1][0] = root['win']
    return plane,label

def loadData(path):
    batch = []
    y = []
    f_list = os.listdir(path)
    for i in f_list:
        try:
            if os.path.splitext(i)[1] == ".json":
                rr = load(i)
                if rr:
                    batch.append(rr[0])
                    y.append(rr[1])
                else:
                    print("data lost")
        except:
            traceback.print_exc()
            continue
    print("end")
    bx = np.asarray(batch)
    by = np.asarray(y)
    return bx,by

# def quick(jsonData):
#     print(jsonData)
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.], [2.]])
#     product = tf.matmul(matrix1, matrix2)
#     sess = tf.Session()
#     result = sess.run(product)
#     print(result)
#     sess.close()
#     return jsonData
if __name__ == "__main__":
    bx, by = loadData(".")
    # cnn.train(bx,by)
    print()


# matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)
    # sess = tf.Session()
    # result = sess.run(product)
    # print result
    # sess.close()
