import socket
import json
import traceback
import NN1 as mynn
import socketserver
import time
import threading
import numpy as np
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
            train = bool(root['train'])
            if not train:
                #fo = open("./test/1.json","w")
                #fo.write(sentenceStr)
                #fo.close()
                cas,mc = mynn.vcnn(train, root, './result/model.ckpt')
                toRet = {}
                toRet["cas"] = cas.tolist()
                toRet["mc"] = float(mc)
                tosend = json.dumps(toRet).encode("utf-8");
                connectionSocket.send(tosend)
                connectionSocket.close()
            else:
                tosend = json.dumps(root).encode("utf-8");
                connectionSocket.send(tosend)
                connectionSocket.close()
                mynn.vcnn(train, './train/', './result/model.ckpt')

        except:
            connectionSocket.close()
            traceback.print_exc()
            continue
if __name__ == '__main__':
    listen()
