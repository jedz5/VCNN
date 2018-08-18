import numpy as np
def xx():
    abc = np.array([0,0,0],dtype=float)
    step = 0.1
    Q = np.array([3,2,1])
    pa = np.array([1,0,0])
    pb = np.array([0, 1, 0])
    pc = np.array([0, 0, 1])
    for i in range(10):
        esp = np.exp(abc)
        pi = esp / np.sum(esp)
        print("pi")
        print(pi)
        action = np.argmax(pi)
        one_hot = np.zeros(3)
        one_hot[action] = 1
        a = (pi*Q*(pa-pi[0])).sum()
        b = (pi * Q * (pb - pi[1])).sum()
        c = (pi * Q * (pc - pi[2])).sum()
        D = step*np.array([a,b,c])
        abc += D
        print("----after delta")
        print(abc)

if __name__ == '__main__':
    xx()