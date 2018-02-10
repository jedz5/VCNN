from socket import *
import json
if __name__ == '__main__':
    serverName = '127.0.0.1'
    serverPort = 50007
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))
    with open("./test/1.json") as inFile:
        root = json.load(inFile)
        sentence = json.dumps(root)
        clientSocket.send(sentence.encode())
        modifiedSentence = clientSocket.recv(1024)
        print('From Server:', modifiedSentence.decode())
    clientSocket.close()