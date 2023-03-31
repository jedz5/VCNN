
from collections import defaultdict, deque
from copy import copy


a = [0,1,2,3,4,5,6,7,8,9]
class Node_inner:
    def __init__(self,i):
        self.val = i
        self.left = None
        self.right = None

def build_full_btree(arr):
    la = len(arr)
    node_list = [Node_inner(i) for i in range(la)]
    for i in range(la):
        a = node_list[i]
        if i*2+1 < la:
            a.left = node_list[i*2+1]
        if i*2+2 < la:
            a.right = node_list[i*2+2]
    return node_list[0]
def build_btree(arr:deque):
    if len(arr) == 0:
        return None
    num = arr.popleft()
    if num < 0:
        return None
    nd = Node_inner(num)
    nd.left = build_btree(arr)
    nd.right = build_btree(arr)
    return nd
def build_btree2(arr,idx):
    # print(f"in = {idx}")
    if len(arr) <= idx:
        print(f"out = [{idx}]")
        return None,idx
    num = arr[idx]
    if num == -1:
        print(f"out = [{idx}]{arr[idx]}")
        return None,idx
    nd = Node_inner(num)
    nd.left,idx_2 = build_btree2(arr,idx+1)
    nd.right,idx_2 = build_btree2(arr,idx_2+1)
    print(f"out = [{idx}]{arr[idx]}")
    return nd,idx_2
def post_tree(node:Node_inner):
    p = node
    st = []
    st.append(p)
    prev = None
    while p.left:
        st.append(p.left)
        p = p.left
    while len(st) > 0:
        p = st.pop()
        if p.right and prev != p.right:
            st.append(p)
            st.append(p.right)
            p = p.right
            while p.left:
                st.append(p.left)
                p = p.left
        else:
            print(p.val)
            prev = p
def isBST(node:Node_inner,l,r):
    if not node:
        return True
    if not (l < node.val < r):
        return False
    t = isBST(node.left,l,node.val)
    if t:
        return isBST(node.right,node.val,r)
    return False

def isBST2(node:Node_inner):
    if node.left:
        t,minl,maxl = isBST2(node.left)
        if not (t and node.val > maxl):
            return False,-1,-1
        nmin = minl
    else:
        nmin = node.val
    if node.right:
        t,minr,maxr = isBST2(node.right)
        if not (t and node.val < minr):
            return False,-1,-1
        nmax = maxr
    else:
        nmax = node.val
    return True,nmin,nmax
def mid_tree(node):
    if not node:
        return
    mid_tree(node.left)
    print(node.val)
    mid_tree(node.right)
def bl_tree(node):
    if not node:
        return True,0
    bl,depth = bl_tree(node.left)
    if bl:
        bl2,depth2 = bl_tree(node.right)
        if bl2:
            if abs(depth - depth2) <= 1:
                return True,max(depth,depth2)
    return False,-1
#[5,-1,6,3,1,-1,2,-1,-1,4,-1,-1,8] [5,-3,-2,-4,-1,6,3,1,-1,2,-1,-1,4,-1,-1,8] [8,6,3,1,-1,2,-1,-1,4,-1,-1,7] Deque([8]) 
# flat_tree = [8,6,3,1,-1,2,-1,-1,4,-1,-1,7]

# r,idx = build_btree2(flat_tree,0)
# mid_tree(r)
# # c = isBST(r,float('-inf'),float('inf'))
# c,nmin,nmax = isBST2(r)
# print(c)

#图的深度优先和广度优先遍历
def DFS(graph,s): #深度优先遍历,基于栈
    stack=[] #建立栈
    stack.append(s) 
    data=[] #记录已经遍历过的点
    data.append(s)
    while stack:
        n=stack.pop()  # 取出栈中最后一个元素并删掉
        nodes=graph[n]
        for i in nodes[::-1]: #栈先进后出
            if i not in data:
                stack.append(i)
                data.append(i)
        print(n)
def dfs2(graph,seen:set,s):
    seen.add(s)
    nexts = graph[s]
    for nd in nexts:
        if nd not in seen:
            dfs2(graph,seen,nd)
    print(s)
def dfs_target(graph,seen:set,s,t):
    print(s)
    if s == t:
        return
    seen.add(s)
    nexts = graph[s]
    for nd in nexts:
        if nd not in seen:
            dfs_target(graph,seen,nd,t)
def BFS(graph,q): #广度优先遍历,基于队列
    queue=[]  #建立队列
    queue.append(q)
    data=[]  #记录已经遍历过的点
    data.append(q)
    while queue:
        n=queue.pop(0)  # 队列先进先出
        nodes=graph[n]
        for j in nodes:
            if j not in data:
                queue.append(j)
                data.append(j)
        print(n)
distij = [[-1] *8] *8
def dij(graph,dist,seen,i,target):
    if i == target:
        return 0
    if (i,target) in seen:
        return 2000
    seen.append((i,target))
    nexts = graph[i]
    seen_copy = copy(seen)
    i_nd_t = [dist[i][nd] + dij(graph,dist,seen_copy,nd,target) for nd in nexts]
    mindij = min(i_nd_t)
    if distij[i][target] == -1:
        distij[i][target] = mindij
    else:
        print(f"reuse distij[{i},{j}]")
    if distij[i][target] > mindij:
        print(f"distij[{i},{j}] {distij[i][target]}>{mindij}")
    return mindij
def swap(arr,i,j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
def partition(arr):
    if len(arr) == 1:
        return arr
    if len(arr) == 2:
        return sorted(arr)
    mid = len(arr) // 2
    swap(arr,mid,len(arr)-1)
    st,ed = 0,len(arr) - 2
    idx = 0
    midx = mid
    pivot = arr[mid]
    for i in range(len(arr)-1):
        if arr[idx] < pivot:
            st += 1
            idx += 1
        elif arr[idx] == pivot:
            pass
class bag_item:
    def __init__(self,w,v,pre) -> None:
        self.w = w
        self.v = v
        self.pre = pre
def max_bag_value(rest_w,arr,idx):
    if idx == len(arr) - 1:
        return 0 
    # if arr[idx].w > rest_w:
    #     return 0
    # if 
    # v1 = arr[idx] *
def choros(arr):
    if len(arr) < 3:
        return False,-1
    if len(arr) == 3:
        if arr[0] < arr[1] and arr[1] > arr[2]:
            return True,3
    ch,lenl = choros(arr[1:])
    if ch:
        if arr[0] < arr[1]:
           return True,lenl+1
    chr,lenr = choros(arr[:-1])
    # if chr:
    #     if arr[-2] > arr[-1]:
    #         return True,
def xx():
    while True:
        try:
            # s = input()
            s = "A Famous Saying: Much Ado About Nothing (2012/8)."
            a = ''
            for i in s:
                if i.isalpha():
                    a += i
            b = sorted(a, key=lambda x:str.upper(x))
            index = 0
            d = ''
            for i in range(len(s)):
                if s[i].isalpha():
                    d += b[index]
                    index += 1
                else:
                    d += s[i]
            print(d)
        except:
            break
def encryp():
    n = ord('a')
    c = chr(n)
    hex()
    bin()
    L1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    L2 = "BCDEFGHIJKLMNOPQRSTUVWXYZAbcdefghijklmnopqrstuvwxyza1234567890"
    # a = defaultdict(chr)
    word_map = dict(zip(L1,L2))
    print(word_map)
def graph_alg():
    # graph={
    #     '1':['2','3'],
    #     '2':['4','5'],
    #     '3':['6','7'],
    #     '4':['8'],
    #     '5':['8'],
    #     '6':['7'],
    #     '7':[],
    #     '8':[],
    #     }
    graph={
        0:[1,2],
        1:[0,3,4],
        2:[0,4,5,6],
        3:[1,7],
        4:[1,2,7],
        5:[2,6],
        6:[2,5],
        7:[3,4],
        }
    dist = [[100,5,3,100,100,100,100,100],
            [5,100,100,8,20,100,100,100],
            [3,100,100,100,1,4,5,100],
            [100,8,100,100,100,1,100,3],
            [100,20,1,100,100,100,100,7],
            [100,100,4,1,100,100,9,100],
            [100,100,5,100,100,9,100,100],
            [100,100,100,3,7,100,100,100],
            ]
    for i in range(8):
        for j in range(8):
            assert dist[i][j] == dist[j][i]
    print("深度优先遍历:")
    # DFS(graph,'1') #结果为1,2,4,8,5,3,6,7
    seen = []
    # dfs2(graph,'1',seen)
    # dfs_target(graph,seen,0,7)
    l = dij(graph,dist,seen,0,7)
    print(l)
    # print("广度优先遍历:")
    # BFS(graph,'1') #结果为1,2,3,4,5,6,7,8
def str_merge():
    while True:
        try:
            in_str = ''.join("eacd ecd".split())
            # str = ''.join(input().split())
        except:
            break
        else:
            str = [0] * len(in_str)
            a = sorted(set(in_str),key=in_str.index)
            str[::2] = sorted(in_str[::2])
            str[1::2] = sorted(in_str[1::2])
            # 第二步奇偶排列
            oushu = [str[x] for x in range(0,len(str),2)]
            jishu = [str[x] for x in range(1,len(str),2)]
            oushu.sort()
            jishu.sort()
            paixu = []
            for i in range(len(str)):
                n = int(i/2)
                if i % 2 == 0:
                    paixu.append(oushu[n])
                else:
                    paixu.append(jishu[n])
            #第三步转换字符
            for i in paixu:
                try:
                    b=bin(int(i,16))[2:]
                    b = '0'*(4-len(b)) + b if len(b)<4 else b
                    b = b[::-1]
                    b = hex(int(b, 2))[2:].upper()
                    print(b,end='')
                except:
                    print(i,end='')
            print()

def str_eval():
    a = eval("1+2*(3+4)")
    print(a)
    "dafsad".count()
def lev_dist():
    str1='oppa' #input()# 等价于
    str2='apple' #input()# 等价于

    # 构建bp表格,str1往str2编辑
    bp=[[x for x in range(len(str2)+1)] for y in range(len(str1)+1)]
    # 现在bp表格中每一行的值为0、1、...len(str2)，其中第0行所有值是正确的
    # 现在要改变第0列的值为0、1、...len(str1)
    for w in range(1,len(str1)):
        bp[w][0]=bp[w-1][0]+1

    # 由于第0行和第0列均更新完成，现在要更新其他空格中的值
    for j in range(1,len(str1)+1):# 从第表格中的第1行第1列，即bp[1][1]处开始，遍历每一行str1:'oppa'
        for k in range(1,len(str2)+1):# 从第表格中的第1行第1列，即bp[1][1]处开始，遍历每一列
            if str1[j-1]==str2[k-1]:# 当最后一个字符相同时，等价于没有改字符
                bp[j][k]=bp[j-1][k-1]
            elif str1[j-1]!=str2[k-1]:#当最后一个字符不相同时，比较左、上、左上三个位置的值，+1后的找最小值，即最小编辑距离
                add=bp[j][k-1]+1
                delete=bp[j-1][k]+1
                replace=bp[j-1][k-1]+1
                bp[j][k]=min(add, delete, replace)

    print(bp[len(str1)][len(str2)])

def commnd_compelte():
    '''
    自己写的用正则匹配太low了，看了@中年美少女写的切片思想佩服了，另外不用字典免去了双层循环的做法，偷来粘在这里供大家欣赏。
    '''
    while True:
        try:
            m="bo a".strip().split() #input()
            key=["reset","reset board","board add","board delete","reboot backplane","backplane abort"]
            value=["reset what","board fault","where to add","no board at all","impossible","install first"]
            #不建字典，用列表的方式避免了双层循环，如果实在要用列表，直接用dict(zip（list1,list2）)合成字典都行.
            if len(m)<1 or len(m)>2:   #判断当输入为小于1个或者输入大于2个字符串时，不符合命令，就报未知命令
                print("unknown command")
            elif len(m)==1:   #当输入一个字符串
                if m[0]==key[0][:len(m[0])]:  #这里才是解决这个题的最佳思想，利用切片的思想来匹配
                    print(value[0])
                else:
                    print("unknown command")
            else:
                index=[]
                for i in range(1,len(key)): #这里把所有原始命令遍历，如果这里写成(len(key)+1),也就是1..6，那么下面的key[i]要改成k[i-1]才符合逻辑
                    a=key[i].split() #将具体的一个KEY分割成两部分
                    if m[0]==a[0][:len(m[0])] and m[1]==a[1][:len(m[1])]:  #然后去匹配被分割的key,这里不可能有reset这种单独的，因为上面条件已经限制了。
                        index.append(i)  #符合条件就把这个位置入列表
                if len(index)!=1:
                    print("unknown command")
                else:
                    print(value[index[0]]) #输出对应的value值
        except:
            break
def local_argmin(dist,visit):
    min = float('inf')
    idx = 0
    for i in range(len(dist)):
        if visit[i] == 0 and min > dist[i]:
            min = dist[i]
            idx = i
    return idx
def dijst():
    graph={
        0:[1,2],
        1:[0,3,4],
        2:[0,4,5,6],
        3:[1,7],
        4:[1,2,7],
        5:[2,6],
        6:[2,5],
        7:[3,4],
        }
    start = 0
    target = 7
    dist = [[100,5,3,100,100,100,100,100],
            [5,100,100,8,20,100,100,100],
            [3,100,100,100,1,4,5,100],
            [100,8,100,100,100,1,100,3],
            [100,20,1,100,100,100,100,7],
            [100,100,4,1,100,100,9,100],
            [100,100,5,100,100,9,100,100],
            [100,100,100,3,7,100,100,100],
            ]
    for i in range(8):
        for j in range(8):
            assert dist[i][j] == dist[j][i]
    dist_res = dist[start][:]
    visit = [0] * len(dist_res)
    visit[start] = 1
    while sum(visit) < len(dist_res):
        selected = local_argmin(dist_res,visit)
        print(selected)
        visit[selected] = 1
        neibours = graph[selected]
        for nb in neibours:
            dist_res[nb] = min(dist_res[selected]+dist[selected][nb],dist_res[nb])
        print(dist_res)
def pretty_print(a,b,dp,zero_end=True):
    if zero_end:
        [print(f" {item}",end=',') for item in b] 
        print()
        for i in range(len(dp)):          
            print(dp[i],end=' ')
            if i==len(dp)-1:
                print()
            else:
                print(a[i])
    else:
        print("  ",end=',')
        [print(f" {item}",end=',') for item in b] 
        print()
        for i in range(len(dp)):          
            print(dp[i],end=' ')
            if i==0:
                print()
            else:
                print(a[i-1])
def common_str():
    a = "ahaabc"
    b = "aahabx"
    dp = [[0] * (len(b)+1) for i in range((len(a)+1))] 
    for i in range(0,len(a)):
        for j in range(0,len(b)):
            if a[i] == b[j]:
                dp[i][j] = dp[i-1][j-1]+1
    pretty_print(a,b,dp)

a = "ahaabc"
b = "aahbabx"
def common_str_recur(i,j,same):
    if i < 0 or j < 0:
        return 0
    if same:
        if a[i] == b[j]:
            return 1+common_str_recur(i-1,j-1,True)
        else:
            return 0
    else:  
        if a[i] == b[j]:
            return max(common_str_recur(i,j-1,False),common_str_recur(i-1,j,False),1+common_str_recur(i-1,j-1,True))
        else:
            return max(common_str_recur(i,j-1,False),common_str_recur(i-1,j,False))

def minDistance(word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if not word1:
            return len(word2)
        if not word2:
            return len(word1)
        dp = [[0 for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for i in range(len(word2) + 1):
            dp[0][i] = i
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        pretty_print(word1,word2,dp,False)
        return dp[len(word1)][len(word2)]
def numDistinct(word1, word2):                                                   
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if not word1:
            return 0
        if not word2:
            return 1
        dp = [[0 for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = 1
        for i in range(len(word2) + 1):
            dp[0][i] = 0
        dp[0][0] = 1
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        pretty_print(word1,word2,dp,False)
        return dp[len(word1)][len(word2)]

def min_coin_count_limited(res,index,money):
    if res == 0:
        return 0
    if index == len(money):
        return 1000000
    value = money[index][0]
    amount = money[index][1]
    





















if __name__ == '__main__':
    r = common_str_recur(len(a)-1,len(b)-1,False)
    print(r)
    