from socket import *
import json

def run_remote_server(serverIP,battle_file):
    serverPort = 50003
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverIP, serverPort))
    with open(battle_file) as inFile:
        root = json.load(inFile)
        root["fileName"] = battle_file
        sentence = json.dumps(root)
        clientSocket.send(sentence.encode(encoding='utf-8'))
        modifiedSentence = clientSocket.recv(1024)
        print('From Server:', modifiedSentence.decode())
    clientSocket.close()
# if __name__ == '__main__':
