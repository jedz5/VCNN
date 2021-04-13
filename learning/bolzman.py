import numpy as np

import matplotlib.pyplot as plt

def coconuts_and_islanders():
    '''椰子与岛民 一共1000个人'''
    num = 30
    '''每个人初始20块钱'''
    money_list = np.zeros((num,),dtype=int) + 3
    rge = 15
    interv = 1
    xmap = np.zeros((rge,),dtype=int)
    '''实验这么多次,每次随机找个人给一块钱给另外一个人'''
    eps = int(10000)
    for ep in range(eps):
        i = np.random.randint(num)
        if money_list[i] > 0:
            money_list[i] -= 1
            money_list[np.random.randint(num)] += 1
        if (ep + 1) % 10000 == 0:
            print(f'实验第{ep}次 {money_list}')
    print("end")
    for x in money_list:
        if x < rge * interv:
            xmap[int(x/interv)] += 1
    # xdata = [f'{x*interv}-{x*interv+interv}' for x in range(rge)]
    xdata = [f'{x*interv}' for x in range(rge)]
    plt.bar(xdata,xmap)
    plt.title("free market ?")
    plt.xlabel("money")
    plt.ylabel("people")
    plt.show()
def free_gas():
    num = 20000
    '''初始都在左边'''
    gas_list = np.zeros((num,), dtype=int)
    rge = 15
    interv = 1
    xmap = np.zeros((rge,), dtype=int)
    '''实验这么多次'''
    eps = int(1000000)
    for ep in range(eps):
        i = np.random.randint(num)
        gas_list[i] = 0 if gas_list[i] else 1
        if (ep + 1) % 10000 == 0:
            print(f'实验第{ep}次 {gas_list.sum()}')
    return gas_list

def coconuts_and_islanders_single_persp():
    num = 100
    '''椰子与岛民 一共100个人,第一个岛民叫小明'''
    '''每个人初始5个椰子'''
    money_list = np.zeros((num,),dtype=int) + 5
    rge = 100
    half_rge = 50
    interv = 10
    xmap = np.zeros((rge,),dtype=int)
    '''实验这么多次,每次随机找个人给一个椰子给另外一个人'''
    eps = int(5000000)
    for ep in range(eps):
        i = np.random.randint(num)
        # if money_list[i] > 0:
        #     money_list[i] -= 1
        #     money_list[np.random.randint(num)] += 1
        # xmap[money_list[0]] += 1
        money_list[i] -= 1
        money_list[np.random.randint(num)] += 1
        if -half_rge*interv < money_list[0] < half_rge*interv:
            xmap[int(money_list[0]/interv)+half_rge] += 1
        if (ep + 1) % 10000 == 0:
            print(f'实验第{ep}次 小明身上有{money_list[0]}个椰子')
    print("end")
    # xdata = [f'{x*interv}-{x*interv+interv}' for x in range(rge)]
    xdata = [f'{(x-half_rge)*interv}' for x in range(rge)]
    plt.bar(xdata,xmap/eps)
    plt.title("free market ?")
    plt.xlabel("XiaoMing's coconut")
    plt.ylabel("percent of the time")
    plt.show()
g = coconuts_and_islanders_single_persp()

