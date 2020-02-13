import socket
import json
import traceback
# import NN1 as mynn
import socketserver
import numpy as np
import os
import platform
from subprocess import Popen
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
# def listen():
#     serverPort = 50007
#     serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     serverSocket.bind(('', serverPort))
#     serverSocket.listen(1)
#     print('The server is ready to receive')
#     while 1:
#         try:
#             connectionSocket, addr = serverSocket.accept()
#             sentence = connectionSocket.recv(40960)
#             print("received ",len(sentence),sentence)
#             sentenceStr = sentence.decode('utf-8')
#             root = json.loads(sentenceStr)
#             train = bool(root['train'])
#             if not train:
#                 #fo = open("./test/1.json","w")
#                 #fo.write(sentenceStr)
#                 #fo.close()
#                 cas,mc = mynn.vcnn(train, root, './result/model.ckpt')
#                 toRet = {}
#                 toRet["cas"] = cas.tolist()
#                 toRet["mc"] = float(mc)
#                 tosend = json.dumps(toRet).encode("utf-8");
#                 connectionSocket.send(tosend)
#                 connectionSocket.close()
#             else:
#                 tosend = json.dumps(root).encode("utf-8");
#                 connectionSocket.send(tosend)
#                 connectionSocket.close()
#                 mynn.vcnn(train, './train/', './result/model.ckpt')
#
#         except:
#             connectionSocket.close()
#             traceback.print_exc()
#             continue
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
    listen_one_battle()
