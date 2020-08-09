import os
import multiprocessing
import random
import json
import time
import numpy as np
import platform
from enum import IntEnum
from subprocess import Popen
from old.client import run_remote_server
Linux = "Linux" == platform.system()
def runClient(ip,port,jsonFile,noGUI="--noGUI"):
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    if Linux:
        clientpath = "./bin/vcmiclient -d --nointro --disable-video --testingport {} --testingfileprefix MPTEST -b {} {}".format(port,jsonFile,noGUI)
        rc = Popen(clientpath)
    else:
        # clientpath = "vcmi_client -d --serverip 192.168.3.200 --nointro --disable-video --testingport {} --testingfileprefix MPTEST -b {}".format(port,jsonFile)
        clientpath = "vcmi_client -d --serverip {} --nointro --disable-video --testingport {} --testingfileprefix MPTEST -b {} {}".format(ip,port,jsonFile,noGUI)
        rc = os.system(clientpath)
    #
    return rc
def runServer(port):
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    if not os.path.exists("train"):
        os.mkdir("train")
    if Linux:
        serverpath = "./bin/vcmiserver -d --port {}".format(port)
    else:
        serverpath = "vcmi_server -d --port {}".format(port)
    # rs = os.system(serverpath)
    rs = Popen(serverpath)
    return rs
def load_json(file):
    with open(file) as JsonFile:
        x = json.load(JsonFile)
        return x
def genJsons(num_samples):
    if Linux:
        os.chdir(r"/home/enigma/rd/share/vcmi/Data")
        crList = load_json(r"/home/enigma/vnn/ENV/creatureData.json")["creatures"]
    else:
        os.chdir(r"D:\project\vcmi\RD\Data")
        crList = load_json(r"D:\project\VCNN\ENV\creatureData.json")["creatures"]
    aa = np.array([1,3,5])
    bb = np.array([aa,aa+14,aa+28,aa+42,aa+56,aa+70,aa+84,aa+98,[119,127,123]])
    creIDs = bb.reshape(-1,)
    samples = []
    for i in range(num_samples):
        # cr1 = random.choice(crList)
        cr11_id = 19
        cr12_id = 15
        cr13_id = 23
        num = 40
        num11 = int(0.7*num)#np.random.randint(15,30)
        num12 = int(1.5*num)#np.random.randint(1,50)
        num13 = 0
        # if random.randint(0,2):
        #     num13 = int(np.random.gamma(2, 2))+1
        cr2_id = 15#random.choice(creIDs)
        cr11 = crList[cr11_id]
        cr12 = crList[cr12_id]
        cr13 = crList[cr13_id]
        cr2 = crList[cr2_id]
        num2 = int(2.5*num)#np.random.randint(20,100)
        def get_enemy_slot_count():
            ai_value1 = cr11["aiValue"] * num11 + cr12["aiValue"] * num12 + cr13["aiValue"] * num13
            ai_value2 = cr2["aiValue"] * num2
            threat = ai_value2 / ai_value1
            split = 1
            if threat <0.5:
                split = 7
            elif threat < 0.67:
                split = 6
            elif threat < 1:
                split = 5
            elif threat < 1.5:
                split = 4
            elif threat < 2:
                split = 3
            else:
                split = 2
            R4 = random.randint(0,100)
            if R4 <= 20:
                split -= 1
            else:
                split += 1
            split = min(split,num2)
            split = min(split, 7)
            return split
        split = get_enemy_slot_count()
        def get_enemy_stacks():
            looseFormat = np.array(
                [[3, 8, 8, 8, 8, 8, 8], [1, 5, 8, 8, 8, 8, 8], [1, 3, 5, 8, 8, 8, 8], [0, 2, 4, 6, 8, 8, 8],
                 [0, 1, 3, 5, 6, 8, 8], [0, 1, 2, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32)
            side1 = np.zeros((7, 2), dtype=np.int32)
            m = int(num2 / split)
            b = split * (m + 1) - num2
            a = split - b
            for i in range(a):
                side1[looseFormat[split - 1, i], 0] = cr2_id
                side1[looseFormat[split - 1, i], 1] = m + 1
            for i in range(a, split):
                side1[looseFormat[split - 1, i], 0] = cr2_id
                side1[looseFormat[split - 1, i], 1] = m
            return side1.tolist()
        def gen_terType():
            BFieldType = IntEnum('BFieldType', 'SAND_SHORE, SAND_MESAS, DIRT_BIRCHES, DIRT_HILLS, DIRT_PINES, GRASS_HILLS,\
            		GRASS_PINES, LAVA, MAGIC_PLAINS, SNOW_MOUNTAINS, SNOW_TREES, SUBTERRANEAN, SWAMP_TREES, FIERY_FIELDS,\
            		ROCKLANDS, MAGIC_CLOUDS, LUCID_POOLS, HOLY_GROUND, CLOVER_FIELD, EVIL_FOG, FAVORABLE_WINDS, CURSED_GROUND,\
            		ROUGH, SHIP_TO_SHIP, SHIP')
            ETerrainType = IntEnum('ETerrainType', 'DIRT, SAND, GRASS, SNOW, SWAMP,\
            		ROUGH, SUBTERRANEAN, LAVA, WATER, ROCK')
            terType = ETerrainType(1 + random.randint(0, 9))
            if terType == ETerrainType.DIRT:
                bfieldType = BFieldType(random.randint(3, 5))
            elif terType == ETerrainType.SAND:
                bfieldType = BFieldType.SAND_MESAS
            elif terType == ETerrainType.GRASS:
                bfieldType = BFieldType(random.randint(6, 7))
            elif terType == ETerrainType.SNOW:
                bfieldType = BFieldType(random.randint(10, 11))
            elif terType == ETerrainType.SWAMP:
                bfieldType = BFieldType.SWAMP_TREES
            elif terType == ETerrainType.ROUGH:
                bfieldType = BFieldType.ROUGH
            elif terType == ETerrainType.SUBTERRANEAN:
                bfieldType = BFieldType.SUBTERRANEAN
            elif terType == ETerrainType.LAVA:
                bfieldType = BFieldType.LAVA
            elif terType == ETerrainType.WATER:
                bfieldType = BFieldType.SHIP
            elif terType == ETerrainType.ROCK:
                bfieldType = BFieldType.ROCKLANDS
            else:
                bfieldType = BFieldType.GRASS_PINES
            return terType.value - 1,bfieldType.value
        terType,bfieldType = gen_terType()
        bat = {}
        bat["bfieldType"] = bfieldType
        bat["terType"] = terType
        bat["sides"] = [{},{}]
        bat["sides"][0]["heroid"] = 24
        # bat["sides"][0]["heroPrimSkills"] = [np.random.randint(0,4),np.random.randint(0,4),1,1]
        bat["sides"][0]["side"] = 0
        if cr2["shoot"] or (np.random.randint(0,3) == 0):
            if num13:
                bat["sides"][0]["army"] = [[cr11_id,num11],[cr13_id,num13],[cr12_id,num12],[15,1],[15,1],[15,1],[15,1]]
            else:
                bat["sides"][0]["army"] = [[cr11_id, num11], [cr12_id,num12],[15,1],[15,1],[15,1],[15,1],[15,1]]
        else:
            if num13:
                bat["sides"][0]["army"] = [[cr11_id,num11],[cr13_id,num13],[cr12_id,num12],[15, np.random.randint(0,2)], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)],
                                           [15, np.random.randint(0,2)]]
            else:
                bat["sides"][0]["army"] = [[cr11_id, num11], [cr12_id, num12], [15, 1], [15, 1], [15, 1], [15, 1],[15, 1]]
                                            #[[cr11_id, num11], [cr12_id, num12], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)],
                                           #[15, np.random.randint(0,2)]]
        bat["sides"][1]["side"] = 1
        bat["sides"][1]["army"] = get_enemy_stacks()
        filename = 'debug.json'.format(cr11["name"].replace(' ',''),num11,cr2["name"].replace(' ',''),num2,i)
        file = open(filename, 'w', encoding='utf-8')
        json.dump(bat,file)
        file.close()
        samples.append(filename)
    return samples
def startBattles():
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    if not os.path.exists("train"):
        os.mkdir("train")
    numCore = os.cpu_count()
    print(numCore)
    port = 7000
    e1 = time.time()
    N = 8
    for j in range(1):
        client_pool = multiprocessing.Pool(processes=N)
        server_pool = multiprocessing.Pool(processes=N)
        client_result = []
        server_result = []
        for i in range(1):
            client_result.append(client_pool.apply_async(runClient, (port + i, "random",)))
            server_result.append(server_pool.apply_async(runServer, (port + i,)))
        client_pool.close()
        server_pool.close()
        client_pool.join()
        server_pool.join()
        err1 = sum([x.get() for x in client_result])
        err2 = sum([x.get() for x in server_result])
        if err1 or err2:
            print("phase1 some wrong")
        print("{} end".format(j))
    e2 = time.time()
    print("开始执行时间：", time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(e1)))
    print("结束执行时间：", time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(e2)))
    print("并行执行时间：", int(e2 - e1))

def start_one_battle(ip,port,remote=False):
    battle = genJsons(1)[0]
    numCore = os.cpu_count()
    print(numCore)
    N = 2
    if remote:
        run_remote_server(ip,battle)
        pass
    else:
        runServer(port)
    runClient(ip,port, battle)
    # runClient(port, "random")
if __name__ == '__main__':
    # genJsons(1)
    start_one_battle("192.168.3.14",3030,False)
