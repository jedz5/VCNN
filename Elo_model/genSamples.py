import os
import platform
import json
import numpy as np
from enum import IntEnum
import random
from old.creatureEmbedding import load_json
from old.creatureEmbedding import run_remote_server
from old.creatureEmbedding import runServer
from old.creatureEmbedding import runClient
import multiprocessing
import glob
import time
import shutil
Linux = "Linux" == platform.system()
def genJsons_one_vs_one(my_army,my_enemy):
    if Linux:
        os.chdir(r"/home/enigma/rd/share/vcmi/Data")
        crList = load_json(r"/home/enigma/vnn/ENV/creatureData.json")["creatures"]
    else:
        os.chdir(r"D:\project\vcmi\RD\Data")
        crList = load_json(r"D:\project\VCNN\ENV\creatureData.json")["creatures"]
    #samples = []
    #for i in range(num_samples):
    def gen_terType():
        BFieldType = IntEnum('BFieldType', 'SAND_SHORE, SAND_MESAS, DIRT_BIRCHES, DIRT_HILLS, DIRT_PINES, GRASS_HILLS,\
                GRASS_PINES, LAVA, MAGIC_PLAINS, SNOW_MOUNTAINS, SNOW_TREES, SUBTERRANEAN, SWAMP_TREES, FIERY_FIELDS,\
                ROCKLANDS, MAGIC_CLOUDS, LUCID_POOLS, HOLY_GROUND, CLOVER_FIELD, EVIL_FOG, FAVORABLE_WINDS, CURSED_GROUND,\
                ROUGH, SHIP_TO_SHIP, SHIP')
        ETerrainType = IntEnum('ETerrainType', 'DIRT, SAND, GRASS, SNOW, SWAMP,\
                ROUGH, SUBTERRANEAN, LAVA, WATER, ROCK')
        terType = ETerrainType.GRASS #ETerrainType(1 + random.randint(0, 9))
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
    bat["sides"][0]["heroPrimSkills"] = [np.random.randint(0,4),np.random.randint(0,4),1,1]
    def breakp():
        pass
    bat["sides"][0]["side"] = 0
    bat["sides"][0]["army"] = my_army
    bat["sides"][1]["side"] = 1
    bat["sides"][1]["army"] = my_enemy
    filename = 'debug.json'
    file = open(filename, 'w', encoding='utf-8')
    json.dump(bat,file)
    file.close()
        #samples.append(filename)
    return filename

def start_one_battle(ip,port,remote,file,nogui):
    numCore = os.cpu_count()
    print(numCore)
    N = 2
    if remote:
        run_remote_server(ip,file)
        pass
    else:
        runServer(port)
    runClient(ip,port, file,nogui)
    # runClient(port, "random")
def startBattles(num,ip,port,file):
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    if not os.path.exists("train"):
        os.mkdir("train")
    numCore = os.cpu_count()
    print(numCore)
    e1 = time.time()
    N = 8
    battle = file
    for j in range(1):
        client_pool = multiprocessing.Pool(processes=N)
        server_pool = multiprocessing.Pool(processes=N)
        client_result = []
        server_result = []
        for i in range(num):
            client_result.append(client_pool.apply_async(runClient, (ip,port + i, battle,"--noGUI",)))
            server_result.append(server_pool.apply_async(runServer, (port + i,)))
        client_pool.close()
        server_pool.close()
        client_pool.join()
        server_pool.join()
        print("{} end".format(j))
    e2 = time.time()
    print("开始执行时间：", time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(e1)))
    print("结束执行时间：", time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(e2)))
    print("并行执行时间：", int(e2 - e1))
def simple_win(path):
    ix=0
    fds = glob.glob(path)
    wins = []
    for inFile in fds:
        with open(inFile) as jsonFile:
            root = json.load(jsonFile)
            wins.append(root["win"])
        ix += 1
    return wins
def start_many_battles(file):
    shutil.rmtree(r'D:\project\vcmi\RD\train')
    os.mkdir(r'D:\project\vcmi\RD\train')
    startBattles(10, "192.168.3.14", 7000,file)
    train_data = simple_win(r'D:\project\vcmi\RD\train\*.json')
    win_rate = (sum(train_data) / len(train_data))
    return win_rate
def count_loss(army,p):
    if np.random.binomial(1,p):
        print("real loss: ",p)
        for cr in army:
            loss = int (cr[1] * p)
            if cr[1] > 0 and loss == 0 and np.random.binomial(1,p):
                loss = 1
            cr[1] -= loss
    else:
        print("win: ",1-p)
    return army
if __name__ == '__main__':
    # genJsons(1)
    # my_army = [[3, 22],[0,1],[0, 1],[0, 1],[0,11],[0,1],[7,12]] #
    # my_enemy = [[23, 5], [23, 5],[23, 5]]
    # start_one_battle("192.168.3.14",8000,False,genJsons_one_vs_one(my_army,my_enemy),'')
    # win_rate =  start_many_battles(genJsons_one_vs_one(my_army))
    # count_loss(my_army,1-win_rate)
    # print("battle1 ",win_rate)
    # print(my_army)
    #
    # win_rate = start_many_battles(genJsons_one_vs_one(my_army))
    # count_loss(my_army, 1 - win_rate)
    # print("battle2 ", win_rate)
    # print(my_army)
    #
    # win_rate = start_many_battles(genJsons_one_vs_one(my_army))
    # count_loss(my_army, 1 - win_rate)
    # print("battle3 ", win_rate)
    # print(my_army)