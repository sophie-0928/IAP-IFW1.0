# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:35:43 2024

@author: 18272022928
"""
import numpy as np
import math
import time
import random
from itertools import combinations
from tsp import solve_tsp
# from pytsp.branch_and_bound_tsp_dfs import branch_and_bound_tsp_bfs

# class TSP:
#     def __init__(self, graph):
#         self.graph = graph
#         self.n = len(graph)
#         self.final_res = float('inf')
#         self.final_path = [None] * (self.n + 1)

#     def first_min_cost(self, i):
#         """Find the minimum edge cost from vertex i."""
#         min_cost = float('inf')
#         for j in range(self.n):
#             if self.graph[i][j] < min_cost and i != j:
#                 min_cost = self.graph[i][j]
#         return min_cost

#     def second_min_cost(self, i):
#         """Find the second minimum edge cost from vertex i."""
#         first, second = float('inf'), float('inf')
#         for j in range(self.n):
#             if i == j:
#                 continue
#             if self.graph[i][j] <= first:
#                 second = first
#                 first = self.graph[i][j]
#             elif(self.graph[i][j] <= second and
#                  self.graph[i][j] != first):
#                 second = self.graph[i][j]
#         return second

#     def bound(self, curr_path, curr_weight, level):
#         """Calculate the lower bound for the current path."""
#         if level == 1:
#             return (self.first_min_cost(curr_path[0]) +
#                     self.first_min_cost(curr_path[1])) / 2

#         if level == 2:
#             return (self.first_min_cost(curr_path[0]) +
#                     self.first_min_cost(curr_path[1]) +
#                     self.first_min_cost(curr_path[2])) / 2

#         return 0

#     def tsp_util(self, curr_bound, curr_weight, level, curr_path):
#         """Recursive utility function to find the minimum cost path."""
#         if level == self.n:
#             if self.graph[curr_path[level - 1]][curr_path[0]] != 0:
#                 curr_res = curr_weight + self.graph[curr_path[level - 1]][curr_path[0]]
#                 if curr_res < self.final_res:
#                     self.final_path[:self.n + 1] = curr_path + [curr_path[0]]
#                     self.final_res = curr_res
#             return

#         for i in range(self.n):
#             if (self.graph[curr_path[level - 1]][i] != 0 and
#                     i not in curr_path):
#                 temp = curr_bound
#                 curr_weight += self.graph[curr_path[level - 1]][i]

#                 if level == 1:
#                     curr_bound -= (self.first_min_cost(curr_path[level - 1]) +
#                                    self.first_min_cost(i)) / 2
#                 else:
#                     curr_bound -= (self.first_min_cost(curr_path[level - 1]) +
#                                    self.first_min_cost(i)) / 2

#                 if curr_bound + curr_weight < self.final_res:
#                     curr_path[level] = i
#                     self.tsp_util(curr_bound, curr_weight, level + 1, curr_path)

#                 curr_weight -= self.graph[curr_path[level - 1]][i]
#                 curr_bound = temp

#     def solve(self):
#         """Find the minimum cost path using the Branch and Bound method."""
#         curr_path = [-1] * (self.n + 1)
#         for i in range(self.n):
#             curr_path[0] = i
#             curr_bound = 0
#             for j in range(self.n):
#                 curr_bound += (self.first_min_cost(i) + self.first_min_cost(j))
#             curr_bound = math.ceil(curr_bound / 2)
#             self.tsp_util(curr_bound, 0, 1, curr_path)
#         return self.final_res, self.final_path

def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2



# Example usage
# if __name__ == "__main__":
#     # graph = [[0, 10, 15, 20],
#     #          [10, 0, 35, 25],
#     #          [15, 35, 0, 30],
#     #          [20, 25, 30, 0]]
#     # tsp = TSP(graph)
#     # result, path = tsp.solve()
#     # print("Minimum cost:", result)
#     # print("Path taken:", path)
run_time=[]
length=[[] for i in range(10)]
seeds=[i for i in range(1,11)]  #种子
S_lst=[250,500,1000,1500,2000]  #前置仓容量
S_lst=[1000,1500,2000]  #前置仓容量
# S_lst=[10] 
# S_lst=[10,20,40,60,80] 
# S_lst=[0]
# seeds=[6]
dist = np.load('dist.npy', allow_pickle=True).tolist()  # 距离矩阵
distance = np.load('distance.npy', allow_pickle=True).tolist()  # 距离矩阵

#分区数量
section=16
#记录每一列的层数
layer=[]
layer.extend([6 for i in range(15)])
layer.extend([1 for i in range(9)])
layer.extend([6 for i in range(6)])
layer.extend([6 for i in range(15)])
layer.extend([1 for i in range(11)])
layer.extend([6 for i in range(4)])

#记录每一个分区的开始列和结束列
# sec_head=[1,10,16,19,23,25,28,30,34,40,43,46,52,55,61,63]
# sec_tail=[9,15,18,22,24,27,29,33,39,42,45,51,54,60,62,66]

sec_head=[1, 7,13,16,20,22,25,27,31,37,40,43,46,49,55,57]
sec_tail=[6,12,15,19,21,24,26,30,36,39,42,45,48,54,56,60]
#每一个分区的销量占比
sales=[0.03,0.06,0.03,0.08,0.04,0.06,0.04,0.12,0.12,0.08,0.05,0.08,0.03,0.12,0.06]


for S in S_lst:
    start=time.process_time()

    for s in seeds:
        random.seed(s)
    # 储位的生成  
    
        # distance=np.load('distance.npy',allow_pickle=True).tolist()  #距离矩阵
        # location=[] #large
        # # location=[i+1 for i in range(goods)] 
        
        # # row=5
        # # column=4        
        # row=25
        # column=20
        # for i in range(column):
        #     for j in range(row):
        #         location.append(i*row+j+1)
        #         location.append(i*row+j+1)
        # random.shuffle(location)
        # location.insert(0, 0)
        
        
        # # 订单的生成
        # # order_n=100
        # # goods=40        
        # order_n=1000
        # goods=1000
        # order=[{} for i in range(order_n)]
        # order_num=[[] for i in range(order_n)]
        # for i in range(order_n):
        #     order_num[i]=random.randint(1,10)
            
        # choices=[i for i in range(1,goods+1)]
        # weight=[]
        # for i in range(1,goods+1):
        #     weight.append(pow(i/goods,0.222))
        # for i in range(goods-1,0,-1):
        #     weight[i]=weight[i]-weight[i-1]
        
        # for i in range(order_n):
        #     temp = random.choices(choices, weight, k=order_num[i])  
        #     temp1=[location[k] for k in temp]
        #     for item in temp1:
        #         order[i][item] = temp1.count(item)
        
        location=[]
        location_t = []  
        
        # 储位的生成
        row = 25
        column = 60
    
        for i in range(column):
            if i+1 in sec_head:
                location_sec=[]
            for j in range(row):
                for k in range(2*layer[i]):
                    location_sec.append(i*row+j+1)
            if i+1 in sec_tail:
                random.shuffle(location_sec)
                location_t.append(location_sec)               

        # 移除并保存元素
        element = location_t.pop(16-1)
        # 在新位置插入元素
        location_t.insert(8, element)
        
        for i in location_t:
            location.extend(i)
        location.insert(0, 0)

        # # 将location按照距离排序
        # location=[]
        # f=[i for i in distance[0]]

        # sku_p=sorted(range(len(f)),key=lambda k: f[k])
        # sku_p.remove(0)
        # for i in sku_p:
        #     location.append(i)
        #     location.append(i)
        # location.insert(0, 0)

        # 订单的生成
        order_n =1000
        goods = len(location)-1
       
        order = [{} for i in range(order_n)]
        order_num = [[] for i in range(order_n)]
        for i in range(order_n):
            order_num[i] = random.randint(1, 10)
            
        choices = [i for i in range(1, goods+1)]
           
    
        #分区生成权重
        weight=[]
        #计算每个区域的商品数
        good_sec=[]
        for i in range(section):
            good_sec.append((sec_tail[i]-sec_head[i]+1)*layer[sec_tail[i]-1]*row*2)
        
        #同一个分区的商品进行合并    
        good_sec[8-1] += good_sec[16-1]
        del good_sec[16-1]
        
        
        for k in range(section-1):  #注意有一个合并操作，所以-1
            weight_sec = []
            if k>=1:
                past=sum(sales[cum] for cum in range(k))
                # print(past)            
            for i in range(1,  good_sec[k]+1):
                weight_sec.append(pow(i/good_sec[k], 0.222))
                
            # # 生成权重
            # for i in range(good_sec[k]-1, 0, -1):
            #     weight_sec[i] = weight_sec[i]-weight_sec[i-1]
            #     weight_sec[i]=sales[k]* weight_sec[i]
            # weight_sec[0]=sales[k]* weight_sec[0]
            # weight.extend(weight_sec)     
          
        #生成累加权重
            for i in range(good_sec[k]):
                weight_sec[i]=sales[k]* weight_sec[i]
                if k>=1:
                    weight_sec[i]=past+ weight_sec[i]
            weight.extend(weight_sec)      
            
        #生成订单
        for i in range(order_n):
            # temp = random.choices(choices, weights=weight, k=order_num[i])  #单品权重，求解速度慢
            temp = random.choices(choices, cum_weights=weight, k=order_num[i]) #累加权重，求解速度快
            
            temp1 = [location[k] for k in temp]
            for item in temp1:
                order[i][item] = temp1.count(item)        
        
            
                
        #组合的计算
        save=[[] for i in range(order_n)] #节省的距离
        w=[[] for i in range(order_n)]  #体积
        comb=[[] for i in range(order_n)]  #组合的信息
        tour=[[] for i in range(order_n)]     #保存路径
        origin=[0 for i in range(order_n)] #原始路径长度
        
        
        
        for i in range(order_n):
            chararray=[]
            for j in order[i].keys():
                chararray.append(j)
            for j in range(1,len(chararray)+1):
                comb[i].extend(combine(chararray, j))                
            chararray.insert(0, 0)
            primgraph = [[0 for col in range(len(chararray))] for row in range(len(chararray))]
            for p in range(len(chararray)-1):
                for q in range(p+1,len(chararray)):
                    primgraph[p][q]=distance[chararray[p]][chararray[q]]
                    primgraph[q][p]=distance[chararray[p]][chararray[q]]
            graph=np.array(primgraph)
            origin[i]=solve_tsp(graph) 
        # print(sum(origin))

            # a.append(0)
            # for j in range(len(a)-1) :
            #     origin[i]=origin[i]+graph[a[j]][a[j+1]]             
            # tour[i]=[chararray[k] for k in a]
            
            # 计算所有组合信息,save和v                        
            for j in comb[i]:
                char=[k for k in chararray]
                for k in list(j):
                    char.remove(k)
                if char==[0]:
                    a=0
                else:
                    primgraph = [[0 for col in range(len(char))] for row in range(len(char))]
                    for p in range(len(char)-1):
                        for q in range(p+1,len(char)):
                            primgraph[p][q]=distance[char[p]][char[q]]
                            primgraph[q][p]=distance[char[p]][char[q]]
                    graph=np.array(primgraph)                
                    a=solve_tsp(graph) 
                
                save[i].append(origin[i]-a)
                temp1=0
                for r in list(j):
                    temp1=temp1+order[i][r]
                w[i].append(temp1)                
 
                           
        #动态规划
        v = [[0 for _ in range(S + 1)] for _ in range(order_n + 1)]
         
        for i in range(1,order_n+1):
            for j in range(1, S + 1):
                v[i][j]=v[i - 1][j]
                for k in range(len(comb[i-1])):
                    if w[i - 1][k] <= j :
                        if v[i - 1][j - w[i-1][k]] + save[i - 1][k] > v[i][j]:
                            v[i][j] = v[i - 1][j - w[i-1][k]] + save[i - 1][k]
                

        # pdr= v[order_n][S]/sum(origin)
        print(sum(origin)-v[order_n][S])
        length[s-1].append(sum(origin)-v[order_n][S])
    
    end=time.process_time()    
    print('Running time:%s Seconds'%(end-start))
    
    run_time.append(end-start)
length= np.array(length)
    # length_mean = np.mean(length,axis=0)
np.save('length',length)
np.save('run_time',run_time)
    