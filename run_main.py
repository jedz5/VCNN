import socket
import json
import traceback
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
            print("utf length ",len(sentenceStr))
            root = json.loads(sentenceStr)
            tosend = json.dumps(root).encode("utf-8");
            connectionSocket.send(tosend)
            connectionSocket.close()
        except:
            connectionSocket.close()
            traceback.print_exc()
            continue
if __name__ == '__main__':
    a = np.asarray([
        [ 0,  0,  0,  0,  0,  0,  0,],
         [ 0,  1,  0,  1,  0,  1,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,]])
    b=np.asarray([
        [ 11,   0,   1,   0,   0,  11,   0,],
         [ 19,   1,   1,   0,  23,   0,   0,],
         [ 19,   0,   1,   0,  23,   0,   0,]])
    c=np.asarray([
        [ 243,    0,   91,   91,    0,   91,    0,],
         [ 274,   91,   91,   91,   91,   91,    0,],
         [ 280,    0,   93,    0,   93,    0,    0,]])
    ac = np.sum(a*c,1)
    bc = np.sum(b*c,1)
    print(ac)
    print(bc)
    print(bc - ac)
