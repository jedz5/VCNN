import socket
import json
import traceback
# import NN1 as mynn
import socketserver
import numpy as np
import os
import platform
from subprocess import Popen
from Elo_model import genSamples
Linux = "Linux" == platform.system()
def runServer_server_side(port=3030):
    if Linux:
        os.chdir("/home/enigma/work/enigma/project/vcmi/RD/builds")
    else:
        os.chdir(r"D:\project\vcmi\RD")
    print(os.getcwd())
    if not os.path.exists("train"):
        os.mkdir("train")
    if Linux:
        serverpath = "./bin/vcmiserver"
    else:
        serverpath = "vcmi_server"
    # rs = os.system(serverpath)
    rs = Popen([serverpath])
    return rs
def gen_quick(root):
    if Linux:
        os.chdir(r"/home/enigma/rd/share/vcmi/Data")
    else:
        os.chdir(r"D:\project\vcmi\RD\Data")
    my_army  = np.zeros([7,2])
    my_enemy = np.zeros([7,2])
    secSkills = []
    for stack in root["stacks"]:
        if stack["isHuman"]:
            my_army[stack["slot"],0] = stack["id"]
            my_army[stack["slot"], 1] = stack["baseAmount"]
        else:
            my_enemy[stack["slot"], 0] = stack["id"]
            my_enemy[stack["slot"], 1] = stack["baseAmount"]
    for sec in root["hero"]["secSkills"]:
        secSkills.append([sec["id"],sec["level"]])
    bat = {}
    bat["bfieldType"] = root["bfieldType"]
    bat["terType"] = root["terType"]
    bat["sides"] = [{}, {}]
    bat["sides"][0]["heroid"] = root["hero"]["id"]
    bat["sides"][0]["heroPrimSkills"] = [root["hero"]["attack"],root["hero"]["defense"],root["hero"]["knowledge"],root["hero"]["power"]]
    bat["sides"][0]["heroSecSkills"] = secSkills
    bat["sides"][0]["side"] = 0
    bat["sides"][0]["army"] = my_army.tolist()
    bat["sides"][1]["side"] = 1
    bat["sides"][1]["army"] = my_enemy.tolist()
    filename = 'runTemp.json'
    file = open(filename, 'w', encoding='utf-8')
    json.dump(bat, file)
    file.close()
    return filename

def listen():
    serverPort = 50007
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(('', serverPort))
    serverSocket.listen(1)
    print('The server is ready to receive')
    while 1:
        try:
            connectionSocket, addr = serverSocket.accept()
            sentence = connectionSocket.recv(40960)
            print("received ",len(sentence),sentence)
            sentenceStr = sentence.decode('utf-8')
            root = json.loads(sentenceStr)
            win_rate = genSamples.start_many_battles(gen_quick(root))
            my_army = [0]*7
            my_enemy = [0] * 7
            win = 1
            for st in root["stacks"]:
                if st["isHuman"]:
                    my_army[st["slot"]] = st["baseAmount"]
                else:
                    my_enemy[st["slot"]] = st["baseAmount"]
            if win_rate >=0.9:
                pass
            elif win_rate > 0:
                p = 1 - win_rate
                for i in range(7):
                    loss = int(my_army[i] * p)
                    if my_army[i] > 0 and loss == 0 and np.random.binomial(1, p):
                        loss = 1
                    my_army[i] -= loss
                if sum(my_army) == 0:
                    win = 0
            else:
                my_army = [0] * 7
                win = 0
            toRet = {}
            toRet["win_rate"] = win_rate
            toRet["cas"] = my_army
            toRet["win"] = win
            print("my_enemy", my_enemy)
            print("my_army", my_army)
            print("win_rate",win_rate)
            print("win", win)
            tosend = json.dumps(toRet).encode("utf-8");
            connectionSocket.send(tosend)
            connectionSocket.close()

        except:
            connectionSocket.close()
            traceback.print_exc()
            continue
def listen_one_battle():
    serverPort = 50003
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(('', serverPort))
    serverSocket.listen(1)
    print('The server is ready to receive')
    while 1:
        try:
            print('accepting...')
            connectionSocket, addr = serverSocket.accept()
            sentence = connectionSocket.recv(40960)
            print("received ",len(sentence),sentence)
            sentenceStr = sentence.decode('utf-8')
            root = json.loads(sentenceStr)
            battle_file = root['fileName']
            if Linux:
                os.chdir(r"/home/enigma/rd/share/vcmi/Data")
                fo = open(battle_file, "w")
                fo.write(sentenceStr)
                fo.close()
            else:
                os.chdir(r"D:\project\vcmi\RD\Data")
            process = runServer_server_side()
            print("start server ok")
            connectionSocket.send("start server ok".encode())
            connectionSocket.close()
        except KeyboardInterrupt:
            exit(0)
        except:
            connectionSocket.close()
            traceback.print_exc()
            # process.terminate()
            continue
if __name__ == '__main__':
    listen()
