# coding:utf-8
# Dijkstra算法――通过边实现松弛
# 指定一个点到其他各顶点的路径――单源最短路径
# 初始化图参数
G = {1: {1: 0, 2: 1, 3: 12}, 2: {2: 0, 3: 9, 4: 3}, 3: {3: 0, 4: 3, 5: 5}, 4: {4: 0, 5: 13, 6: 15}, 5: {5: 0, 6: 4},
     6: {6: 0}}


# 每次找到离源点最近的一个顶点，然后以该顶点为重心进行扩展
# 最终的到源点到其余所有点的最短路径
# 一种贪婪算法


def n_short(G, n, INIT=999, max_n=3):
    # 递归实现n_short路径的寻找，返回每个节点的代价与路径
    if n == 1:
        path = {1: {0: [[0], ]}}
        return path
    else:
        point = n
        n = n - 1
        pre_path = n_short(G, n)
        #print(pre_path)
        cost = {}
        if G[0].get(point, 'NULL') != 'NULL':
            cost.update({G[0][point]:[[0],]})
        for node, cost_dict in pre_path.items():
            if G[node].get(point, 'NULL') != 'NULL':
                max_cost = sorted(list(cost_dict.keys()))
                # print(max_cost)
                for i in range(len(max_cost)):
                    if i + 1 > max_n:
                        pass
                        # break
                    if (G[node].get(point, 'NULL') + max_cost[i]) not in cost:
                        cost[G[node].get(point, 'NULL') + max_cost[i]] = []
                    for each_path in cost_dict[max_cost[i]]:
                        # print(each_path)
                        # print(cost[G[node].get(point, 'NULL') + max_cost[i]])
                        cost[G[node].get(point, 'NULL') + max_cost[i]].append(each_path + [node])
        pre_path.update({point: cost})
        return pre_path


'''
def n_short(G, v0, INIT=999, max_n=3):
    book = set()
    dis = dict((k, INIT) for k in G.keys())
    path = {1:{0:[1]}}
    dis[v0] = 0
    minv = v0
    while max(path.keys()) < max(G.keys()):
        point = max(path.keys())+1
        for node, path in path.items():
            if path == {}:
                path[node].update({0:})
        book.add(minv)
        for w in G[minv]:
            if dis[minv] + G[minv][w] < dis[w]:
                dis[w] = dis[minv] + G[minv][w]
        path.append(minv)
        new = INIT
        for v in dis.keys():
            if v in book: continue
            if dis[v] < new:
                new = dis[v]
                minv = v
    pass

    return dis, path
'''


def Dijkstra(G, v0, INF=999, max_n=3):
    """ 使用 Dijkstra 算法计算指定点 v0 到图 G 中任意点的最短路径的距离
    INF 为设定的无限远距离值
    此方法不能解决负权值边的图
    """
    book = set()
    minv = v0
    # 源顶点到其余各顶点的初始路程
    dis = dict((k, INF) for k in G.keys())
    dis[v0] = 0
    while len(book) < len(G):
        book.add(minv)  # 确定当期顶点的距离
        for w in G[minv]:  # 以当前点的中心向外扩散
            if dis[minv] + G[minv][w] < dis[w]:  # 如果从当前点扩展到某一点的距离小与已知最短距离
                dis[w] = dis[minv] + G[minv][w]  # 对已知距离进行更新
        new = INF  # 从剩下的未确定点中选择最小距离点作为新的扩散点
        for v in dis.keys():
            if v in book: continue
            if dis[v] < new:
                new = dis[v]
                minv = v
    return dis


dis = Dijkstra(G, v0=1)
print("测试结果：")
print(dis.values())


def P(string):
    pass
    return 0.8


import re
import data_preparation as dp

train_c = dp.main('train')
most_common = train_c.most_common(1200)
patt = [count[0] for count in most_common]
patt = '|'.join(patt)
patt = re.compile(patt)

most_common_set = set([])
for i in most_common:
    most_common_set.add(i[0])

test_string = 'بېكىتەلمەيدىغانلىقى'

def get_vocab(most_common=None, filename='train'):
    train_c = dp.main(filename)
    if not most_common:
        most_common = len(train_c.keys())
    most_common = train_c.most_common(most_common)

    most_common_set = set([])
    for i in most_common:
        most_common_set.add(i[0])

    return most_common_set

def g_build(string):
    g = {}
    n = len(string)
    for i, s in enumerate(string):
        g[i] = {i: 0}
        if i + 1 != n:
            g[i].update({i + 1: 1})
        for j in range(n - i + 1):
            if string[i:i+j] in most_common_set:
                print(string[i:i+j])
                g[i].update({i+j: P(string[i:i+j])})
    g[n] = {n: 0}
    return g


