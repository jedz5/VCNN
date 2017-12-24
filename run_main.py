import socket
import json
import traceback
import socketserver
import time
import threading

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