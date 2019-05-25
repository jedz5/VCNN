import os
import multiprocessing
import random
import json
import time
def runClient(port,jsonFile):
    clientpath = "./bin/vcmiclient -d --nointro --disable-video --noGUI --testingport {} --testingfileprefix MPTEST -b {}".format(port,jsonFile)
    rc = os.system(clientpath)
    return rc
def runServer(port):
    serverpath = "./bin/vcmiserver -d --port {}".format(port)
    rs = os.system(serverpath)
    return rs
def load_json(file):
    with open(file) as JsonFile:
        x = json.load(JsonFile)
        return x
def genJsons(num_samples):
    os.chdir(r"D:\project\VCNN\ENV\RD\Data\battlejson")
    crList = load_json(r"D:\project\VCNN\ENV\creatureData.json")["creatures"]
    samples = []
    for i in range(num_samples):
        cr1 = random.choice(crList)
        cr2 = random.choice(crList)
        bat = {}
        bat["bfieldType"] = 2
        bat["terType"] = 2
        bat["sides"] = [{},{}]
        bat["sides"][0]["side"] = 0
        bat["sides"][0]["army"] = [[cr1["id"],0]] * 7
        bat["sides"][1]["side"] = 1
        bat["sides"][1]["army"] = [[cr2["id"], 0]] * 7
        slot1 = random.randint(0,6)
        slot2 = random.randint(0,6)
        hp = random.randint(300,3000)
        num1 = max(1,int(hp/cr1["hp"]))
        num2 = max(1, int(hp / cr2["hp"]))
        bat["sides"][0]["army"][slot1] = [cr1["id"],num1]
        bat["sides"][1]["army"][slot2] = [cr2["id"],num2]
        filename = '{}-{}-{}-{}-{}.json'.format(cr1["name"].replace(' ',''),num1,cr2["name"].replace(' ',''),num2,i)
        file = open(filename, 'w', encoding='utf-8')
        json.dump(bat,file)
        samples.append(filename)
    return samples
def startBattles():
    os.chdir("/home/enigma/work/enigma/project/vcmi/RD/install")
    numCore = os.cpu_count()
    print(numCore)
    port = 30000
    e1 = time.time()
    N = 4
    for j in range(1):
        client_pool = multiprocessing.Pool(processes=N)
        server_pool = multiprocessing.Pool(processes=N)
        client_result = []
        server_result = []
        for i in range(2000):
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
if __name__ == '__main__':
    # genJsons(5000)
    startBattles()