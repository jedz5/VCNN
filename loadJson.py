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
spells = {'26': 0, '41': 1, '53': 2, '54': 3}
def loadTest(root):
    plane = np.zeros((side,stacks,hexDepth))
    plane_m = np.zeros((n_manaCost))
    #addPlane = np.ndarray((side,stacks,addDepth),dtype=np.dtype('a20'))
    #for i in range(side):
     #   for j in range(stacks):
     #       for k in range(addDepth):
     #           addPlane[i][j][k] = ''
    try:
        if not 'hero' in root:
            return
        heroStrength = np.math.sqrt((1 + 0.05 * root['hero']['attack']) * (1 + 0.05 * root['hero']['defense']))
        for x in root['stacks']:
            plane[~x['isHuman']][x['slot']][0] = x['baseAmount']
            plane[~x['isHuman']][x['slot']][1] = x['aiValue']
            plane[~x['isHuman']][x['slot']][2] = isFly(x)
            plane[~x['isHuman']][x['slot']][3] = isShoot(x)
            plane[~x['isHuman']][x['slot']][4] = x['speed']
            plane[~x['isHuman']][x['slot']][5] = x['health']
            #addPlane[~x['isHuman']][x['slot']][0] = x['name']
            if x['isHuman']:
                plane[~x['isHuman']][x['slot']][1] = x['aiValue'] * heroStrength  #baseAD
                #archary
                if isShoot(x):
                    plane[~x['isHuman']][x['slot']][1] = plane[~x['isHuman']][x['slot']][1]*shootDamage(root['hero'])
                else: #offence
                    plane[~x['isHuman']][x['slot']][1] = plane[~x['isHuman']][x['slot']][1] * meleeDamage(root['hero'])
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
    plane[-1] = root['hero']['mana']
    plane_m[0] = root['hero']['mana']

    return plane,plane_m,origPlane
def loadTrain(inFile):
    with open(inFile) as jsonFile:
        root = json.load(jsonFile)
        plane = np.zeros((side,stacks,hexDepth))
        plane_m = np.zeros((n_manaCost))
        label_c = np.zeros((stacks))
        label_m = np.zeros((n_manaCost))
        addPlane = np.ndarray((side,stacks,addDepth),dtype=np.dtype('a20'))
        for i in range(side):
            for j in range(stacks):
                for k in range(addDepth):
                    addPlane[i][j][k] = ''
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

    return plane,plane_m,label_c,label_m,origPlane,addPlane
def loadTestData(inJson):
    batchx = []
    batchm = []
    origPlane = []
    try:
        rr = loadTest(inJson)
        if rr:
            batchx.append(rr[0])
            batchm.append(rr[1])
            origPlane.append(rr[2])
        else:
            print("data lost")
            return
    except:
        traceback.print_exc()
        return
    bx = np.asarray(batchx)
    bxm = np.asarray(batchm)
    borigPlane = np.asarray(origPlane)
    return bx,bxm,borigPlane
def loadTrainData(path):
    batchx = []
    batchm = []
    yc = []
    ym = []
    origPlane = []
    addPlane = []
    f_list = os.listdir(path)
    for i in f_list:
        try:
            if os.path.splitext(i)[1] == ".json":
                rr = loadTrain(path+i)
                if rr:
                    batchx.append(rr[0])
                    batchm.append(rr[1])
                    yc.append(rr[2])
                    ym.append(rr[3])
                    origPlane.append(rr[4])
                    addPlane.append(rr[5])
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
    borigPlane = np.asarray(origPlane)
    baddPlane = np.asarray(addPlane)
    return bx,bxm,byc,bym,borigPlane,baddPlane
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
def storeTrainSimple(jsonsPath):
    fils = os.listdir(jsonsPath)
    NSamples = 60000
    data = np.zeros([min(len(fils), NSamples), 10 * 14], dtype=int)
    label = np.zeros([min(len(fils), NSamples), 7], dtype=int)

    i = -1
    for a in fils:
        i += 1
        if i >= NSamples:
            break
        with open(jsonsPath+a) as s:
            stackLocation = {}
            root = json.load(s)
            for st in root["stacks"]:
                slot = 10 * (st["slot"] + (0 if st["isHuman"] else 7))
                data[i, slot] = st["id"] + 1
                data[i, slot + 1] = st["baseAmount"]
                data[i, slot + 2] = st["speed"]
                data[i, slot + 3] = st["luck"]
                data[i, slot + 4] = st["morale"]
                data[i, slot + 5] = st["attack"]
                data[i, slot + 6] = st["maxDamage"]
                data[i, slot + 7] = st["minDamage"]
                data[i, slot + 8] = st["health"]
                data[i, slot + 9] = st["defense"]
                if st["isHuman"]:
                    if st["id"] not in stackLocation:
                        stackLocation[st["id"]] = st["slot"]
                    label[i,[stackLocation[st["id"]]]] += st["killed"]
    np.save("./dataset/samples56.npy",data,label)
if __name__ == "__main__":
    data = storeTrainSimple("./samples56/train/")