# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:28:47 2024

@author: 18272022928
"""

import numpy as np
import math
import time
import random
from itertools import combinations
from tsp import solve_tsp

seeds = [i for i in range(1,11)]  # 种子
S_list = [250, 500, 1000, 1500, 2000]  # 前置仓容量
# S_list=[10,20,40,60,80] 
# S_list = [250]
# seeds = [1]
pdr_list = [[] for i in range(10)]
ratio=[[] for i in range(10)]
origin_list=[[] for i in range(10)]
route_list=[[] for i in range(10)]
reduction=[[] for i in range(10)]
run_time=[]
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


for S1 in S_list:
    start=time.process_time()
    for s in seeds:
        random.seed(s)
        location=[]
        location_t = []  
        
        # 储位的生成
        row = 25
        column = 60
        lot=row*column
        
        f1=[0 for j in range(lot)] # 主干道的距离
        for j in range(lot):    
            f1[j] = dist[lot+int(column/2)][j] 
        f1.insert(0, 0)    
        
        
        for i in range(column):
            if i+1 in sec_head:
                location_sec=[]
            for j in range(row):
                for k in range(2*layer[i]):
                    location_sec.append(i*row+j+1)
            if i+1 in sec_tail:
                random.shuffle(location_sec)
                
                # f=[distance[0][d] for d in location_sec] #按距离排序location
                # sku_p=sorted(range(len(f)),key=lambda k: f[k]) 
                # location_sec=[location_sec[d] for d in sku_p]    
                
                # f=[f1[d] for d in location_sec] #按距离排序location
                # sku_p=sorted(range(len(f)),key=lambda k: f[k],reverse=True) 
                # location_sec=[location_sec[d] for d in sku_p]                
                
                location_t.append(location_sec)               

        # 移除并保存元素
        element = location_t.pop(16-1)
        # 在新位置插入元素
        location_t.insert(8, element)
        
        for i in location_t:
            location.extend(i)
        location.insert(0, 0)


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
            temp2 = random.choices(choices, cum_weights=weight, k=order_num[i]) #累加权重，求解速度快
            
            temp1 = [location[k] for k in temp2]
            for item in temp1:
                order[i][item] = temp1.count(item)

        
        # # 计算距离
        # d_0=0
        # for i in range(order_n):
        #     for j in order[i]:
        #         d_0+=distance[0][j]*order[i][j]


        # 组合的计算
        u = [[] for i in range(order_n)]  # 单位价值
        U = []  # 每个组合中最大的单位价值
        I = []  # 每个组合中最大的单位价值组合索引
        save = [[] for i in range(order_n)]  # 节省的距离
        v = [[] for i in range(order_n)]  # 体积
        com = [[] for i in range(order_n)]  # 组合的信息
        tour = [[] for i in range(order_n)]  # 保存路径
        origin = [0 for i in range(order_n)]  # 原始路径长度

        for i in range(order_n):
            chararray = [0]
            for j in order[i].keys():
                chararray.append(j)
            primgraph = [[0 for col in range(len(chararray))]
                         for row in range(len(chararray))]
            for p in range(len(chararray)-1):
                for q in range(p+1, len(chararray)):
                    primgraph[p][q] = distance[chararray[p]][chararray[q]]
                    primgraph[q][p] = distance[chararray[p]][chararray[q]]
            graph = np.array(primgraph)
            
            # a = christofides_tsp(graph)
            # a.append(0)
            a=solve_tsp(graph)
            for j in range(len(a)-1):
                origin[i] = origin[i]+graph[a[j]][a[j+1]]
            tour[i] = [chararray[k] for k in a]
       
            
            # 开始选组合
            for j in range(len(chararray)-1, 0, -1):  # 组合个数 n+...+1
                long = len(chararray)-j  # 组合长度
                for k in range(j):
                    temp = 0
                    for r in range(long+1):
                        temp = temp+graph[a[k+r]][a[k+r+1]]
                    temp = temp-graph[a[k]][a[k+r+1]]
                    save[i].append(temp)
                    temp1 = 0
                    for r in range(long):
                        temp1 = temp1+order[i][tour[i][k+r+1]]
                    v[i].append(temp1)
                    com[i].append([k+1, long])
            for j in range(len(v[i])):
                u[i].append(save[i][j]/v[i][j])
            U.append(max(u[i]))
            I.append(u[i].index(max(u[i])))

        # 循环筛选
        P_e = 0
        P_bf = 0
        S=S1
        d_f=0
        while (S > 0):
            m = U.index(max(U))  # 确认哪个订单里的组合
            C = com[m][I[m]]  # 根据组合的索引找到组合的信息
            S = S-v[m][I[m]]
            if S >= 0:
                for i in range(C[1]):
                    
                    del tour[m][C[0]]
                                   
                P_bf = P_bf+save[m][I[m]]
                if tour[m] == [0, 0]:
                    U[m] = 0
                else:
                    save[m] = []
                    u[m] = []
                    v[m] = []
                    com[m] = []
                    for j in range(len(tour[m])-2, 0, -1):  # 组合个数 n+...+1
                        long = len(tour[m])-j-1  # 组合长度
                        for k in range(j):
                            temp = 0
                            for r in range(long+1):
                                temp = temp + \
                                    distance[tour[m][k+r]][tour[m][k+r+1]]
                            temp = temp-distance[tour[m][k]][tour[m][k+r+1]]
                            save[m].append(temp)
                            temp1 = 0
                            for r in range(long):
                                temp1 = temp1+order[m][tour[m][k+r+1]]
                            v[m].append(temp1)
                            com[m].append([k+1, long])
                    for j in range(len(v[m])):
                        u[m].append(save[m][j]/v[m][j])
                    U[m] = max(u[m])
                    I[m] = u[m].index(max(u[m]))
            else:
                P_e = P_e+save[m][I[m]]
                C_e = []
                for i in range(C[1]):
                    C_e.append(tour[m][C[0]+i])

        # 比较
        route = [0 for i in range(order_n)]
        if P_e < P_bf:
            # 最终路径长度
            for i in range(order_n):
                for j in range(len(tour[i])-1):
                    route[i] = route[i]+distance[tour[i][j]][tour[i][j+1]]
            pdr = 1-sum(route)/sum(origin)
            print(pdr)
        else:
            for j in C_e:
                tour[m].remove(j)
            for j in range(len(tour[m])-1):
                route[m] = route[m]+distance[tour[m][j]][tour[m][j+1]]
            pdr = 1-sum(route[m])/sum(origin[m])
            print(pdr)

        pdr_list[s-1].append(pdr)
        # ratio[s-1].append((d_f/S1)/(d_0/5500.0))
        origin_list[s-1].append(sum(origin))
        reduction[s-1].append(sum(origin)-sum(route))
        route_list[s-1].append(sum(route))
        print(sum(route))   
    end=time.process_time()    
    # print('Running time:%s Seconds'%(end-start))    
    run_time.append(end-start)
    print(run_time)
pdr_list = np.array(pdr_list)
route_list = np.array(route_list)
column_sum = np.mean(pdr_list,axis=0)

# np.save("a",pdr_list)
# ratio= np.array(ratio)
# ratio_mean = np.mean(ratio,axis=0)
# origin_list= np.array(origin_list)
# origin_mean = np.mean(origin_list,axis=0)
# reduction= np.array(reduction)
# reduction_mean = np.mean(reduction,axis=0)