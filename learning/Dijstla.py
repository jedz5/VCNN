import numpy as np


inf = 10086
def dijstra(start: int, mgraph: list) -> list:
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]
    dis = mgraph[start]

    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]:
                idx = i

        nopass.remove(idx)
        passed.append(idx)

        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + mgraph[idx][i]
    return dis

def bellman(start: int, mgraph: list):
    mgraph = np.array(mgraph,dtype=int)
    dist = np.zeros((len(mgraph)),dtype=int)
    dist[:] = mgraph[start,:]
    dist_new = np.zeros((len(mgraph)), dtype=int)
    dist_new[:] = mgraph[start, :]
    while True:
        print("new iter")
        for i in range(len(mgraph)):
            dist_new[i] = min(dist[:]+mgraph[:,i]) #起点s到顶点i的最短距离 <--- min(s到i周围点k的最短距离+k到i的距离)
            print(dist_new)
        if (dist_new - dist).sum() == 0:
            break
        else:
            dist[:] = dist_new[:]
# def water_flow(start: int, mgraph: list):
#     mgraph = np.array(mgraph, dtype=int)
#     nodes = [start]
#     dist = np.zeros((len(mgraph)), dtype=int)
#     while len(nodes):



if __name__ == "__main__":
    mgraph = [[0, 1, 12, inf, inf, inf],
              [inf, 0, 9, 3, inf, inf],
              [inf, inf, 0, inf, 5, inf],
              [inf, inf, 4, 0, 13, 15],
              [inf, inf, inf, inf, 0, 4],
              [inf, inf, inf, inf, inf, 0]]

    dis = bellman(0, mgraph)
