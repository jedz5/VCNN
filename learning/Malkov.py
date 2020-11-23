import numpy as np
import copy
import matplotlib.pyplot as plt
import pdb
np.set_printoptions(precision=3,suppress=True)
from scipy.special import comb, perm
def malkov_plot():
    import mpl_toolkits.axisartist as axisartist
    fig = plt.figure(figsize=(2, 2))
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    # 通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis[:].set_visible(False)

    # ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    # 给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size=1.0)
    # 添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # 设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")

    # transfer_matrix = np.array([[0.5,0.25,0.25],[0.5,0,0.5],[0.25,0.25,0.5]],dtype='float32').transpose()
    transfer_matrix = np.array([[0.8, 0.2], [0.3, 0.7]], dtype='float32').transpose()
    tt = np.array([1, 0])
    xs, ys = [tt[0]], [tt[1]]
    for i in range(10):
        tt = np.matmul(transfer_matrix, tt)
        print(tt)
        xs.append(tt[0])
        ys.append(tt[1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(xs, ys)
    plt.show()
''' 
    y = Ax
    A = PVP^
    x' = P^x
    y' = Vx'
    y = Py'
'''
def tranverse(p,pi,count):
    for i in range(count):
        pi = np.dot(p,pi)
    return pi

def get_stationary_prob(transfer_matrix):
    pi = np.zeros((transfer_matrix.shape[0],1))
    a, P = np.linalg.eig(transfer_matrix)
    pi[0, 0] = 0
    pi[1, 0] = 1
    pi2 = x_x2(P, pi)
    stationary_index = a.argmax(axis=0)
    tmp = pi2[stationary_index, 0]
    pi2[:, 0] = 0
    pi2[stationary_index, 0] = tmp
    pi = y2_y(P, pi2)
    return pi
def random_wakl_absorb():
    transfer_matrix = np.zeros((10, 10))
    transfer_matrix[0, 0] = 1
    transfer_matrix[9, 9] = 1
    for i in range(1, 9):
        transfer_matrix[i, [i - 1, i + 1]] = 1 / 2
    transfer_matrix = transfer_matrix.transpose()
    a, P = np.linalg.eig(transfer_matrix)
def random_wakl_revert():
    transfer_matrix = np.zeros((10, 10))
    transfer_matrix[0, 0] = 0.9
    transfer_matrix[0, 1] = 0.1
    transfer_matrix[9, 9] = 0.9
    transfer_matrix[9, 8] = 0.1
    for i in range(1, 9):
        transfer_matrix[i, [i - 1, i + 1]] = 1 / 2
    transfer_matrix = transfer_matrix.transpose()
    p = get_stationary_prob(transfer_matrix)
    print(p)
    pi = np.zeros((10,))
    pi[3] = 1
    p = tranverse(transfer_matrix,pi,500)
    print(p)
    pdb.set_trace()
def idea_gas():
    NUM = 11
    transfer_matrix = np.zeros((NUM, NUM))
    pi = np.zeros((NUM, 1))
    pi2 = np.zeros((NUM, 1))
    transfer_matrix[0, 1] = 1
    transfer_matrix[NUM - 1, NUM - 2] = 1
    for i in range(1, NUM-1):
        transfer_matrix[i, i + 1] = 1 - i/10
        transfer_matrix[i, i - 1] = i / 10

    return a,P
def x_x2(P,x):
    x2 = np.dot(np.linalg.inv(P), x)
    return x2
def y2_y(P,y2):
    y = np.dot(P, y2)
    return y
def map_comb_nature():
    count = 0
    all = 8
    index = np.zeros((all,))
    for i in range(2,all):
        for j in range(1,i):
            for k in range(j):
                # print()
                count +=1
    print(count)
def get_neibs(t,a_i):
    nebs = {}
    i, j, k, l = t
    a = f'{i}-{j}-{k}-{l}'
    self_index = a_i[a]
    nebs[self_index] = 0
    for idi in range(len(t)):
        x = t[idi]
        if x == 0:
            nebs[self_index] += 1/4
        else:
            for idj in range(len(t)):
                if idi != idj:
                    tups = copy.copy(t)
                    tups[idi] -= 1
                    tups[idj] += 1
                    i,j,k,l = tups
                    a = f'{i}-{j}-{k}-{l}'
                    id = a_i[a]
                    nebs[id] = 1/12
    return nebs
def coconuts_and_islanders():
    count = 0
    N = 12
    # #计算排列数
    # A=perm(3,2)
    # #计算组合数
    # C=comb(45,2)
    # print(A,C)
    size = int(comb(N + 3, 3))
    a_i = {}
    i_a = np.zeros((size,), dtype=object)
    tranverse_P = np.zeros((size, size))
    for i in range(N + 1):
        j_left = N - i
        for j in range(j_left + 1):
            k_left = j_left - j
            for k in range(k_left + 1):
                tmp = f'{i}-{j}-{k}-{k_left-k}'
                print(tmp)
                a_i[tmp] = count
                i_a[count] = tmp
                count += 1
    if size != count:
        print(f"count={count} != size={size}")
        exit(-1)
    for i in range(N + 1):
        j_left = N - i
        for j in range(j_left + 1):
            k_left = j_left - j
            for k in range(k_left + 1):
                tmp = f'{i}-{j}-{k}-{k_left-k}'
                self_index = a_i[tmp]
                nb = get_neibs([i, j, k, k_left - k], a_i)
                for x, y in nb.items():
                    tranverse_P[self_index, x] = y

    transfer_matrix = tranverse_P.transpose()
    p = get_stationary_prob(transfer_matrix)
    print(p)
    pdb.set_trace()
M = 0
if __name__ == '__main__':
    # coconuts_and_islanders()
    X = np.array([[.3, .5, .2], [.5, .1, .4], [.2, .4, .4]])
    p = get_stationary_prob(X)