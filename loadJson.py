import json
import os
import traceback
import numpy as np
import sys
print(sys.path)
side = 2
stacks = 7
hexDepth = 2
n_spells = 4
n_manaCost = 1
n_all = side*stacks*hexDepth+n_spells+n_manaCost
def load(inFile):
    root = None
    with open(inFile) as jsonFile:
        root = json.load(jsonFile)
        plane = np.zeros((side,stacks,hexDepth))
        plane_m = np.zeros((n_manaCost));
        label_c = np.zeros((stacks))
        label_m = np.zeros((n_manaCost));
        amout = np.zeros((stacks))
        value = np.zeros((stacks))
        spells = {'26':1,'41':2,'53':3,'54':4}
        try:
            if not 'hero' in root:
                return
            heroStrength = np.math.sqrt((1 + 0.05 * root['hero']['attack']) * (1 + 0.05 * root['hero']['defense']))
            for x in root['stacks']:
                plane[~x['isHuman']][x['slot']][0] = x['baseAmount']
                #plane[~x['isHuman']][x['slot']][1] = x['fightValue']
                plane[~x['isHuman']][x['slot']][1] = x['aiValue']
                if x['isHuman']:
                    #plane[~x['isHuman']][x['slot']][1] = x['fightValue'] * heroStrength
                    plane[~x['isHuman']][x['slot']][1] = x['aiValue'] * heroStrength
                    amout[x['slot']] = x['baseAmount']
                    value[x['slot']] = x['aiValue'] * heroStrength
                    label_c[x['slot']] = x['killed']
            plane.resize((n_all,),refcheck=False)
            if 'spells' in root['hero']:
                for y in root['hero']['spells']:
                    if str(y['id']) in spells:
                        plane[spells[str(y['id'])]+n_all-5 -1] = 1

        except:
            traceback.print_exc()
            return
        plane[-1] = root['manaCost']
        plane_m[0] = root['hero']['mana']
        label_m[0] = root['manaCost']

    return plane,plane_m,label_c,label_m,amout,value

def loadData(path):
    batchx = []
    batchm = []
    yc = []
    ym = []
    batch_amout = []
    batch_value = []
    f_list = os.listdir(path)
    for i in f_list:
        try:
            if os.path.splitext(i)[1] == ".json":
                rr = load(i)
                if rr:
                    batchx.append(rr[0])
                    batchm.append(rr[1])
                    yc.append(rr[2])
                    ym.append(rr[3])
                    batch_amout.append(rr[4])
                    batch_value.append(rr[5])
                else:
                    print("data lost")
        except:
            traceback.print_exc()
            continue
    print("end")
    bx = np.asarray(batchx)
    bxm = np.asarray(batchm)
    byc = np.asarray(yc)
    bym = np.asarray(ym)
    b_amount = np.asarray(batch_amout)
    b_value = np.asarray(batch_value)
    return bx,bxm,byc,bym,b_amount,b_value

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
    bx, bxm, byc, bym, b_amount,b_value = loadData(".")
    mPercent = bym /bxm
    print()


# matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)
    # sess = tf.Session()
    # result = sess.run(product)
    # print result
    # sess.close()
