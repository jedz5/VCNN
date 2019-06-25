import os
import multiprocessing
import random
import json
import time
import numpy as np
import platform
from subprocess import Popen
Linux = "Linux" == platform.system()
def runClient(port,jsonFile):
    if Linux:
        clientpath = "./bin/vcmiclient -d --nointro --disable-video --noGUI --testingport {} --testingfileprefix MPTEST -b {}".format(port,jsonFile)
        rc = Popen(clientpath)
    else:
        clientpath = "vcmi_client -d --serverip 127.0.0.1 --nointro --disable-video --testingport {} --testingfileprefix MPTEST -b {}".format(port,jsonFile)
        rc = os.system(clientpath)
    #
    return rc
def runServer(port):
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
    bb = np.array([aa,aa+14,aa+28,aa+42,aa+56,aa+70,aa+84,aa+98])
    creIDs = bb.reshape(-1,)
    samples = []
    for i in range(num_samples):
        # cr1 = random.choice(crList)
        cr11_id = 19
        cr12_id = 15
        cr13_id = 23
        num11 = int(np.random.gamma(2, 2)) + 10
        num12 = int(10*np.random.randn()) + 20
        num12 = max(1,num12)
        num13 = 0
        if random.randint(0,2):
            num13 = int(np.random.gamma(2, 2))+1
        cr2_id = random.choice(creIDs)
        cr11 = crList[cr11_id]
        cr12 = crList[cr12_id]
        cr13 = crList[cr13_id]
        cr2 = crList[cr2_id]
        num2 = int(10*np.random.randn()) + 50
        num2 = max(1,num2)
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
        bat = {}
        bat["bfieldType"] = 2
        bat["terType"] = 2
        bat["sides"] = [{},{}]
        bat["sides"][0]["heroid"] = 24
        bat["sides"][0]["heroPrimSkills"] = [np.random.randint(0,4),np.random.randint(0,4),1,1]
        bat["sides"][0]["side"] = 0
        if num13:
            bat["sides"][0]["army"] = [[cr11_id,num11],[cr13_id,num13],[cr12_id,num12],[15,1],[15,1],[15,1],[15,1]]
        else:
            if (cr2["shoot"] or np.random.randint(0,2)):
                bat["sides"][0]["army"] = [[cr11_id, num11], [cr12_id, num12], [15, 1], [15, 1], [15, 1], [15, 1],
                                       [15, 1]]
            else:
                bat["sides"][0]["army"] = [[cr11_id, num11], [cr12_id, num12], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)], [15, np.random.randint(0,2)],
                                           [15, np.random.randint(0,2)]]
        bat["sides"][1]["side"] = 1
        bat["sides"][1]["army"] = get_enemy_stacks()
        filename = '{}-{}-{}-{}-{}.json'.format("grandElf",25,cr2["name"].replace(' ',''),num2,i)
        file = open(filename, 'w', encoding='utf-8')
        json.dump(bat,file)
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

def start_one_battle():
    battle = genJsons(1)[0]
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    if not os.path.exists("train"):
        os.mkdir("train")
    numCore = os.cpu_count()
    print(numCore)
    port = 7000
    N = 2
    runServer(port)
    runClient(port, battle)
if __name__ == '__main__':
    # genJsons(20)
    startBattles()
