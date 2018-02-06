import json
import os
import traceback
import numpy as np
import sys
print(sys.path)
side = 2
stacks = 7
hexDepth = 6
n_spells = 5
n_manaCost = 1
addDepth = 1
n_all = side*stacks*hexDepth+n_spells+n_manaCost
def isFly(x):
    if "ability" in x:
        for y in x['ability']:
            if y['type'] == 43:
                return 1
    return 0
def isShoot(x):
    if "ability" in x:
        for y in x['ability']:
            if y['type'] == 44:
                return 1
    return 0
def noRetaliate(x):
    if "ability" in x:
        for y in x['ability']:
            if y['type'] == 68:
                return 1
    return 0
def shootDamage(hero):
    if "secSkills" in hero:
        for ss in hero['secSkills']:
            if ss['id'] == 1:
                return 1+ ss['level']*0.1
    return 1
def meleeDamage(hero):
    for ss in hero['secSkills']:
        if ss['id'] == 22:
            return 1+ ss['level']*0.6
    return 1
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
        addPlane = np.ndarray((side,stacks,addDepth),dtype=np.dtype('a20'))
        for i in range(side):
            for j in range(stacks):
                for k in range(addDepth):
                    addPlane[i][j][k] = ''
        spells = {'26':0,'41':1,'53':2,'54':3}
        try:
            if not root['win']:
                print(inFile)
                return
            if not 'hero' in root:
                return
            if root['quickBattle']:
                return
            heroStrength = np.math.sqrt((1 + 0.05 * root['hero']['attack']) * (1 + 0.05 * root['hero']['defense']))
            for x in root['stacks']:
                plane[~x['isHuman']][x['slot']][0] = x['baseAmount']
                plane[~x['isHuman']][x['slot']][1] = x['aiValue']
                plane[~x['isHuman']][x['slot']][2] = isFly(x)
                plane[~x['isHuman']][x['slot']][3] = isShoot(x)
                plane[~x['isHuman']][x['slot']][4] = x['speed']
                plane[~x['isHuman']][x['slot']][5] = x['health']
                addPlane[~x['isHuman']][x['slot']][0] = x['name']
                if x['isHuman']:
                    plane[~x['isHuman']][x['slot']][1] = x['aiValue'] * heroStrength  #baseAD
                    #archary
                    if isShoot(x):
                        plane[~x['isHuman']][x['slot']][1] = plane[~x['isHuman']][x['slot']][1]*shootDamage(root['hero'])
                    else: #offence
                        plane[~x['isHuman']][x['slot']][1] = plane[~x['isHuman']][x['slot']][1] * meleeDamage(root['hero'])
                    amout[x['slot']] = x['baseAmount']
                    value[x['slot']] = plane[~x['isHuman']][x['slot']][1]
                    label_c[x['slot']] = x['killed']
            origPlane = plane.copy()
            plane.resize((n_all,),refcheck=False)
            if 'spells' in root['hero']:
                for y in root['hero']['spells']:
                    if str(y['id']) in spells:
                        plane[spells[str(y['id'])]+n_all-n_spells-n_manaCost] = 1
                    else:
                        plane[-2] = 1  #other magic

        except:
            traceback.print_exc()
            return
        plane[-1] = root['hero']['mana'] + root['manaCost']
        plane_m[0] = root['hero']['mana'] + root['manaCost']
        label_m[0] = root['manaCost']

    return plane,plane_m,label_c,label_m,amout,value,origPlane,addPlane

def loadData(path):
    batchx = []
    batchm = []
    yc = []
    ym = []
    batch_amout = []
    batch_value = []
    origPlane = []
    addPlane = []
    f_list = os.listdir(path)
    for i in f_list:
        try:
            if os.path.splitext(i)[1] == ".json":
                rr = load(path+i)
                if rr:
                    batchx.append(rr[0])
                    batchm.append(rr[1])
                    yc.append(rr[2])
                    ym.append(rr[3])
                    batch_amout.append(rr[4])
                    batch_value.append(rr[5])
                    origPlane.append(rr[6])
                    addPlane.append(rr[7])
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
    borigPlane = np.asarray(origPlane)
    baddPlane = np.asarray(addPlane)
    return bx,bxm,byc,bym,b_amount,b_value,borigPlane,baddPlane
def trans(fromF,toF):
    toRoot = {}
    with open(fromF) as jsonFile:
        root = json.load(jsonFile)
        if "terType" in root:
            toRoot["terType"] = root["terType"]
        if "bfieldType" in root:
            toRoot["bfieldType"] = root["bfieldType"]
        toRoot["mana"] = root["hero"]["mana"]
        toRoot["sides"] = []
        me = {}
        me["side"] = 0
        if "heroid" in root["hero"]:
            me["heroid"] = root["hero"]["heroid"]
        else:
            me["heroid"] = 99
        me["heroPrimSkills"] = [root["hero"]["attack"], root["hero"]["defense"], root["hero"]["knowledge"], root["hero"]["power"]]
        me["heroSecSkills"] = []
        me["spells"] = []
        me["army"] = []
        for seck in root["hero"]["secSkills"] :
            me["heroSecSkills"].append([seck["id"],seck["level"]])
        for stacks in root["stacks"] :
            if stacks["isHuman"]:
                me["army"].append([stacks["id"],stacks["baseAmount"]])
        if 'spells' in root['hero']:
            for sp in root['hero']['spells']:
                me["spells"].append(sp["id"])
        toRoot["sides"].append(me)

        you = {}
        you["side"] = 1
        you["army"] = []
        for stacks in root["stacks"]:
            if not stacks["isHuman"]:
                you["army"].append([stacks["id"], stacks["baseAmount"]])
        toRoot["sides"].append(you)
        with open(toF,'w') as outF:
            json.dump(toRoot,outF)
if __name__ == "__main__":
    trans("./train/br-0-20180206T221729.json","./train/br-0-20180206T221729-trans.json")
