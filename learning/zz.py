from collections import defaultdict
def f():
    n = int(input().strip())
    n += 1
    in_str = input().strip() #"dasdbuiodevauufgh"
    # in_str = "aeueo"
    aeiou = "aeiouAEIOU"
    in_str = ['z' if ch in aeiou else 'x' for ch in in_str ]
    while in_str[-1] == 'x':
        in_str.pop()
        if len(in_str) == 0:
            print(0)
            return
    in_str = in_str[::-1]
    while in_str[-1] == 'x':
        in_str.pop()
    in_str = in_str[::-1]
    in_str_sp = "".join(in_str).split('x')
    in_str_sp = [len(sp) for sp in in_str_sp]
    sum_ls = []
    for idx in range(len(in_str_sp) - n +1):
        if in_str_sp[idx] == 0:
            continue
        else:
            sum_len = sum(in_str_sp[idx:idx+n])
            sum_ls.append(sum_len)
    if len(sum_ls) > 0:
        print(max(sum_ls))
    else:
        print(0)
f()



in_str = "xyxyXX"
# in_str = "abababb"
while True:
    try:
        in_str = input()
        ch_count = defaultdict(int)
        for ch in in_str:
            ch_count[ch] += 1
        res = sorted(ch_count,key=lambda x:ch_count[x],reverse=True)
        for ch in res:
            print(f"{ch}:{ch_count[ch]}",end=';')
    except:
        break



m,n = map(int,input().strip().split())
tasks = list(map(int,input().strip().split()))
# m,n = map(int,"3 5".strip().split())
# tasks = list(map(int,"8 4 3 2 10".strip().split()))
tasks.sort(reverse=True)
work_lines = [0] * m
tik = 0
if n <= m:
    print(max(tasks))
else:
    for i in range(m):
        work_lines[i] = tasks.pop()
    while True:
        tik += 1
        for i in range(m):
            work_lines[i] -= 1
            if work_lines[i] <= 0:
                if len(tasks) > 0:
                    work_lines[i] = tasks.pop()
                else:
                    work_lines[i] = 0
        if sum(work_lines) == 0:
            print(tik)
            break
