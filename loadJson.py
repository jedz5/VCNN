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
        plane_m = 0;
        label_c = np.zeros((stacks))
        label_m = 0
        hero = np.zeros((n_spells))
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
                    label_c[x['slot']] = x['killed']
            #plane.resize((n_all,),refcheck=False)
            if 'spells' in root['hero']:
                for y in root['hero']['spells']:
                    if str(y['id']) in spells:
                        hero[spells[str(y['id'])]-1] = 1

        except:
            traceback.print_exc()
            return
        plane_m = root['hero']['mana']
        label_m = root['manaCost']
        #label[1][0] = root['win']
    return plane,plane_m,label_c,label_m,hero

def loadData(path):
    batchx = []
    batchm = []
    yc = []
    ym = []
    hero = []
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
                    hero.append(rr[4])
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
    bh = np.asarray(hero)
    return bx,bxm,byc,bym,bh

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
    bx, bxm, byc, bym, bh = loadData(".")
    b_amount = np.copy(bx[:, 0, :, 0])
    b_value = np.copy(bx[:, 0, :, 1])
    np.resize(bx, (len(bx), n_all), refCheck=False)
    for i in (len(bx)):
        for j in range(4):
            bx[i][side * stacks * hexDepth + j] = bh[i][j]
        bx[i][-1] = bxm[i]
    mPercent = bxm / bym
    print()


# matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)
    # sess = tf.Session()
    # result = sess.run(product)
    # print result
    # sess.close()
