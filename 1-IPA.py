# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:46:08 2024

@author: 18272022928
"""

import random
import numpy as np
from scipy.optimize import linprog
import copy
import time 
import math
# distance_matrix=[[0,1,1.4,1],[1,0,1,1.4],[1.4,1,0,1],[1,1.4,1,0]]
start=time.process_time()
from pytsp.christofides_tsp import christofides_tsp
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from collections import OrderedDict

def merge_dicts_with_sum(dicts):
    merged = defaultdict(int)  # 使用int初始化，这样可以直接进行加法
    for d in dicts:
        for key, value in d.items():
            merged[key] += value
    return dict(merged)  # 返回普通字典
def golden_section_search(a, b, tol=1e-1):
    golden_ratio = (math.sqrt(5) - 1) / 2
    length = b - a

    x1 = a + (1 - golden_ratio) * length
    x2 = a + golden_ratio * length
    
    while x2-x1>tol:
       
        x11=[x[i]+x1*d[i] for i in range(goods)]
        x22=[x[i]+x2*d[i] for i in range(goods)]
        if fmin (order,x11) < fmin (order,x22):
            b = x2
            x2 = x1
            x1 = a + (1 - golden_ratio) * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + golden_ratio * (b - a)

    return a

def tour (chararray):
    length=0
    primgraph = [[0 for col in range(len(chararray))] for row in range(len(chararray))]
    for p in range(len(chararray)-1):
        for q in range(p+1,len(chararray)):
            primgraph[p][q]=distance[chararray[p]][chararray[q]]
            primgraph[q][p]=distance[chararray[p]][chararray[q]]
    graph=np.array(primgraph)
    a=christofides_tsp(graph) 
    a.append(0)
    for k in range(len(a)-1) :
        length=length+graph[a[k]][a[k+1]]             
         
    return length


    
def IPA (x,j):
    # good=[]
   
    # for j in range(goods) :
    #     if x[j]!=0:
    #         good.append(j+1)
    # #消耗掉x的库存，计算此时的f   
    # for j in good :
    for d in range(days):
        inventory=0
        for i in range(order_n):
            if j in order[d][i]:
                inventory=inventory+order[d][i][j]
                if inventory<=x[j-1]:
                    del order[d][i][j]
                if inventory>=x[j-1]:
                    break
            
    #计算每种商品的增量（每天增量求均值）
    f = [[0 for d in range(days)]  for j in range(goods)]
    for j in range(1,goods+1) :
    # for j in range(2,3) :
        for d in range(days):
            inventory=0
            for i in range(order_n):
                if j in order[d][i]:
                    inventory=inventory+order[d][i][j]
                    if inventory<=1:
                        #计算原路径长度
                        site={}
                        temp1=[location[k] for k in order[d][i] ]
                        for item in temp1:
                            site[item] = temp1.count(item)    
                        chararray=[0]
                        for k in site:
                            chararray.append(k)   
                        # print(chararray)
                        origin=tour (chararray)
                        # print(origin)
                        #删除操作后的路径长度
                        temp1=[k for k in order[d][i] ]
                        
                        temp1.remove(j)
                       
                        site={}
                        temp2=[location[k] for k in temp1 ]
                        for item in temp2:
                            site[item] = temp2.count(item)    
                        chararray=[0]
                        for k in site:
                            chararray.append(k)  
                        # print(chararray)
                        new=tour (chararray)
                        # print(new)
                        f[j-1][d]=new-origin
                        # print(f[0][0])
                        break
                    else:
                        break
    gradient=[]
    for j in range(goods):
        gradient.append(sum(f[j])/days)                
    return gradient

def fmin (order,x):
    good=[]
    order=copy.deepcopy(order)
    for j in range(goods) :
        if x[j]!=0:
            good.append(j+1)
    #消耗掉x的库存，计算此时的f   
    for j in good :
        for d in range(days):
            inventory=0
            for i in range(order_n):
                if j in order[d][i]:
                    inventory=inventory+order[d][i][j]
                    if inventory<=x[j-1]:
                        del order[d][i][j]
                    if inventory>=x[j-1]:
                        break
    # 计算现在的距离
    #计算距离
    order2=[[{} for i in range(order_n)] for j in range(days)]  #以储位记录的订单,计算距离时只需记录访问节点，无需明确商品数量
    route=[[0 for i in range(order_n)] for d in range(days) ]
    for d in range(days):  
        for i in range(order_n):
            temp1=[location[k] for k in order[d][i]]
            for item in temp1:
                order2[d][i][item] = temp1.count(item)
    for d in range(days):  
        for i in range(order_n):
            chararray=[0]
            for j in order2[d][i].keys():
                chararray.append(j)
            primgraph = [[0 for col in range(len(chararray))] for row in range(len(chararray))]
            for p in range(len(chararray)-1):
                for q in range(p+1,len(chararray)):
                    primgraph[p][q]=distance[chararray[p]][chararray[q]]
                    primgraph[q][p]=distance[chararray[p]][chararray[q]]
            graph=np.array(primgraph)
            a=christofides_tsp(graph) 
            a.append(0)
            for j in range(len(a)-1) :
                route[d][i]=route[d][i]+graph[a[j]][a[j+1]]  
    return(sum(sum(route[d]) for d in range(days))/days)







S_list=[250,500,1000,1500,2000]

# S_list=[1,2,3,4,5]
# S_lst=[5]
x_list=[]
days=10
dist = np.load('dist.npy', allow_pickle=True).tolist()  # 距离矩阵
distance=np.load('distance.npy',allow_pickle=True).tolist()  #距离矩阵
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
#每个location隶属的通道
aisle=[]
for i in range (60):
    for k in range(25):
        aisle.append(i+1)


# #train
# random.seed(11)

# # 储位的生成  
# location=[]
# location_t = []  

# # 储位的生成
# row = 25
# column = 60
# lot=row*column

# f1=[0 for j in range(lot)] # 主干道的距离
# for j in range(lot):    
#     f1[j] = dist[lot+int(column/2)][j] 
# f1.insert(0, 0) 

# for i in range(column):
#     if i+1 in sec_head:
#         location_sec=[]
#     for j in range(row):
#         for k in range(2*layer[i]):
#             location_sec.append(i*row+j+1)
#     if i+1 in sec_tail:
#         random.shuffle(location_sec)
        
#         # f=[distance[0][d] for d in location_sec] #按距离排序location
#         # sku_p=sorted(range(len(f)),key=lambda k: f[k]) 
#         # location_sec=[location_sec[d] for d in sku_p]  
        
#         # f=[f1[d] for d in location_sec] #按距离排序location
#         # sku_p=sorted(range(len(f)),key=lambda k: f[k],reverse=True) 
#         # location_sec=[location_sec[d] for d in sku_p]
        
#         location_t.append(location_sec)               

# # 移除并保存元素
# element = location_t.pop(16-1)
# # 在新位置插入元素
# location_t.insert(8, element)

# for i in location_t:
#     location.extend(i)
# location.insert(0, 0)

# np.save("location.npy",location)

# # 订单的生成
# order_n =1000
# goods = len(location)-1
    
# choices = [i for i in range(1, goods+1)]
   
# #分区生成权重
# weight=[]
# #计算每个区域的商品数
# good_sec=[]
# for i in range(section):
#     good_sec.append((sec_tail[i]-sec_head[i]+1)*layer[sec_tail[i]-1]*row*2)

# #同一个分区的商品进行合并    
# good_sec[8-1] += good_sec[16-1]
# del good_sec[16-1]


# for k in range(section-1):  #注意有一个合并操作，所以-1
#     weight_sec = []
#     if k>=1:
#         past=sum(sales[cum] for cum in range(k))
#         # print(past)            
#     for i in range(1,  good_sec[k]+1):
#         weight_sec.append(pow(i/good_sec[k], 0.222))            
  
# #生成累加权重
#     for i in range(good_sec[k]):
#         weight_sec[i]=sales[k]* weight_sec[i]
#         if k>=1:
#             weight_sec[i]=past+ weight_sec[i]
#     weight.extend(weight_sec)      
    

# order=[[{} for i in range(order_n)] for j in range(days)]
# for d in range(days):
#     order_num=[0 for i in range(order_n)]
#     for i in range(order_n):
#         order_num[i]=random.randint(1,10)
    
#     for i in range(order_n):
#         temp = random.choices(choices, cum_weights=weight, k=order_num[i]) #累加权重，求解速度快 
#         for item in temp:
#             order[d][i][item] = temp.count(item)       


# #统计各商品出现的频次
# D=[]
# for d in range(days):          
#     D.append(merge_dicts_with_sum(order[d]))
# # result=merge_dicts_with_sum(result_list)
# for d in range(days): 
#     for i in range(1,goods+1):
#         if i not in D[d]:
#             D[d][i]=0
            
# #DFW算法
# K=1000   #迭代次数
# delta=0.1  #终止条件
# x=[0 for i in range(goods)]  
# j=0  #IPA最小的商品索引,初值为0

# # #构造初始解-1
# # for kk in range(S_list[-1]):
# #     G=IPA(x,j)     
# #     #贪心算法构造初始值
# #     j=G.index(min(G))
# #     x[j]+=1
# #     j=j+1  #索引转换
# #     if kk+1 in S_list:
# #         print(kk+1)
# #         temp=[i for i in x]
# #         x_list.append(temp)
        
# iteration=0 
# x_list1=[] 
# for S in S_list:
   
#     #构造初始解-2
#     m=gp.Model("wo")
#     m.setParam('MIPGap', 0.0)
#     Q=m.addVars(goods,vtype=GRB.CONTINUOUS, name="Q")
#     u=m.addVars(goods,days,vtype=GRB.CONTINUOUS, name="u")
    
#     m.addConstr((sum(Q[i] for i in range(goods))<=S),"con1")
#     m.addConstrs((Q[i]>=0 for i in range(goods)),"con2")  
#     m.addConstrs((u[i,j]>=D[j][i+1]-Q[i]  for j in range(days) for i in range(goods)),"con3")
#     m.addConstrs((u[i,j]>=0 for i in range(goods) for j in range(days)),"con4")
#     m.setObjective((1/days)*sum(distance[0][location[i+1]]*u[i,j] for i in range(goods) for j in range(days)),GRB.MINIMIZE)    
    
#     m.optimize()
#     Q= {ind:Q[ind].x for ind in Q}    
   
#     temp=[0 for i in range(goods)]
#     for i in range(goods):
#         temp[i]=Q[i]
#     x_list.append(temp)
   
#     # 加载初始解
#     x=x_list[iteration]
    
#     #迭代
#     w = [1 for i in range(goods)]
#     for kk in range(K):
#         G=IPA(x,j)    
#         c=[G[i] for i in range(goods)] 
#         A_ub=[w]
#         b_ub=[S]
#         x_bounds=[(0,None) for i in range(goods)]
#         result = linprog(c, A_ub, b_ub, bounds=x_bounds, method='simplex')
#         s=list(result.x)
        
        
#         d=[s[i]-x[i] for i in range(goods)]
#         g=-sum(G[i]*d[i] for i in range(goods))
#         if g<0.1:
#             break
#         else:
#             # alpha=2/(S+k)  #迭代1
#             alpha = golden_section_search(0, 1)  #迭代2
#             x=[x[i]+alpha*d[i] for i in range(goods)]
            
#     iteration+=1         
#     x_list1.append(x)
    

# end=time.process_time()
# print(end-start) 
# np.save("x_0.npy", x_list1) 
# # np.save("x_1.npy", x_list1) 
# # np.save("x_2.npy", x_list1) 


# test
x_list=np.load('x_0.npy',allow_pickle=True).tolist()
# x_list=np.load('x_1.npy',allow_pickle=True).tolist() #按距离排序
# x_list=np.load('x_2.npy',allow_pickle=True).tolist() #按exposure排序 


seeds=[i for i in range(1,11)]  #种子
l0=[[] for i in range(10)]
pdr_list=[[] for i in range(10)]
ratio=[[] for i in range(10)]
aisle_list=[[] for i in range(10)]

# 初始距离,不同布局对应的距离不同
origin = np.load('origin.npy', allow_pickle=True).tolist()
# origin = np.load('origin1.npy', allow_pickle=True).tolist()
# origin = np.load('origin2.npy', allow_pickle=True).tolist()  

nonstop=[i for i in range(16,31)]
iteration=0
for S in S_list:
    s1=0
    for s in seeds:
        random.seed(s)
        
        # 储位的生成  
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
                #以下分别是三种布局
                random.shuffle(location_sec)
                
                # f=[distance[0][d] for d in location_sec] #按距离排序location
                # sku_p=sorted(range(len(f)),key=lambda k: f[k]) 
                # location_sec=[location_sec[d] for d in sku_p]    
                
                # f=[f1[d] for d in location_sec] #按exposure排序location
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
        location = np.load('location.npy', allow_pickle=True).tolist()

        # 订单的生成
        order_n =1000
        goods = len(location)-1
        aisle_num=0
        orders = [{} for i in range(order_n)]
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
            for item in temp:
                orders[i][item] = temp.count(item) 
                
            #统计每个订单的原始通道个数    
            temp3 = [aisle[location[k]-1] for k in temp]
            temp4=list(set(temp3))
            aisle_num+=len(temp4)
            if temp3[0]  not in nonstop:
                aisle_num+=1
            if temp3[-1]  not in nonstop:
                aisle_num+=1
        # 计算距离
        d_0=0
        for i in range(order_n):
            for j in orders[i]:
                d_0+=distance[0][location[j]]*orders[i][j]
                
        d_f=0    
        for j in range(goods) :
            d_f+=x_list[iteration][j]*distance[0][location[j]]
                
        # ratio[s-1].append((d_f/S)/(d_0/5500.0))
        
        #消耗掉x的库存，计算此时的f 
        good=[]
        for j in range(goods) :
            if x_list[iteration][j]!=0:
                good.append(j+1)
        
        for j in good :
            inventory=0
            for i in range(order_n):
                if j in orders[i]:
                    inventory=inventory+orders[i][j]
                    if inventory<=x_list[iteration][j-1]:
                        del orders[i][j]
                    if inventory>=x_list[iteration][j-1]:
                        break
                        
        #计算距离
        order=[{} for i in range(order_n)]  #以储位记录的订单,计算距离时只需记录访问节点，无需明确商品数量
        route=[0 for i in range(order_n)]   
        aisle_num_1=0
        for i in range(order_n):
            temp1=[location[k] for k in orders[i]]
            for item in temp1:
                order[i][item] = temp1.count(item)
 
            #统计每个订单还需进入的通道个数    
            temp5 = [aisle[k-1] for k in temp1]
            temp6=list(set(temp5))
            aisle_num_1+=len(temp6)                   
            if len(temp5)>0:
                if temp5[0]  not in nonstop:
                    aisle_num_1+=1
                if temp5[-1]  not in nonstop:
                    aisle_num_1+=1                 
                
                
        for i in range(order_n):
            chararray=[0]
            for j in order[i].keys():
                chararray.append(j)
            primgraph = [[0 for col in range(len(chararray))] for row in range(len(chararray))]
            for p in range(len(chararray)-1):
                for q in range(p+1,len(chararray)):
                    primgraph[p][q]=distance[chararray[p]][chararray[q]]
                    primgraph[q][p]=distance[chararray[p]][chararray[q]]
            graph=np.array(primgraph)
            # route[i]=solve_tsp(graph) 
            # if order[i]=={}:
            #     route[i]=0
            a=christofides_tsp(graph) 
            a.append(0)
            for j in range(len(a)-1) :
                route[i]=route[i]+graph[a[j]][a[j+1]]    
                
        # print(sum(route))        
        
        pdr=1-sum(route)/origin[s-1] 
        print(pdr)
        pdr_list[s-1].append(pdr)  
        aisle_list[s-1].append((aisle_num-aisle_num_1)/aisle_num)                 
                          
    iteration+=1                        
pdr_list=np.array(pdr_list)  
pdr_sum = np.mean(pdr_list,axis=0) 
aisle_list=np.array(aisle_list)
aisle_mean=np.mean(aisle_list,axis=0)  
np.save("a",pdr_list)  
