# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:58:22 2024

@author: 18272022928
"""

import random
import numpy as np
# distance_matrix=[[0,1,1.4,1],[1,0,1,1.4],[1.4,1,0,1],[1,1.4,1,0]]
# from tsp import solve_tsp
from pytsp.christofides_tsp import christofides_tsp


seeds=[i for i in range(1,11)]  #种子

# #初始距离代码
# S_lst=[0] 
# origin=[]

#求PDR代码
S_lst=[250,500,1000,1500,2000]  #前置仓容量
origin = np.load('origin2.npy', allow_pickle=True).tolist()  # 初始距离




l0=[[] for i in range(10)]
pdr_list=[[] for i in range(10)]



ratio=[[] for i in range(10)]

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
sec_head=[1, 7,13,16,20,22,25,27,31,37,40,43,46,49,55,57]
sec_tail=[6,12,15,19,21,24,26,30,36,39,42,45,48,54,56,60]
#每一个分区的销量占比
sales=[0.03,0.06,0.03,0.08,0.04,0.06,0.04,0.12,0.12,0.08,0.05,0.08,0.03,0.12,0.06]
# sales=[0.03,0.10,0.04,0.08,0.04,0.06,0.04,0.07,0.12,0.08,0.05,0.08,0.03,0.12,0.06]
# sales=[0.03,0.06,0.03,0.08,0.04,0.06,0.04,0.12,0.08,0.12,0.05,0.08,0.03,0.12,0.06]

for S in S_lst:
    for s in seeds:
        random.seed(s)
   
    # 储位的生成  
        location=[]
        location_t = []  
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
                random.shuffle(location_sec)  #随机location

                f=[distance[0][d] for d in location_sec] #按距离排序location
                sku_p=sorted(range(len(f)),key=lambda k: f[k]) 
                location_sec=[location_sec[d] for d in sku_p]  
                
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
       
        orders=[{} for i in range(order_n)]  #以商品记录的订单
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






        # # 计算距离
        # d_0=0
        # for i in range(order_n):
        #     for j in orders[i]:
        #         d_0+=distance[0][location[j]]*orders[i][j]    
        
        
        
        # #这是一段附加代码，要对流行商品进行删除       
        # # 生成权重
        # weight_ind=[0 for i in range(goods)]
        # for i in range(goods-1, 0, -1):
        #     weight_ind[i] = weight[i]-weight[i-1]
        # weight_ind[0]=weight[0]
        # #排序
        # sku_p=sorted(range(len(weight_ind)),key=lambda k: weight_ind[k],reverse=True) 

        # popular=0    
        # for j in sku_p:
        #     for i in range(order_n):    
        #         if j in orders[i]:
        #             popular=popular+orders[i][j]
        #             if popular<=S:
        #                 del orders[i][j]
        #             if popular>=S:
        #                 break
        #     if (popular>=S) :        
        #         break    
        
        
        # # 这是一段附加代码，要对远距离商品进行删除 
        # d_f=0
        # f=[i for i in distance[0]]
        # s_count=0
        # sku_p=sorted(range(len(location)),key=lambda k: f[location[k]],reverse=True) 
        # for j in sku_p:
        #     for i in range(order_n):    
        #         if j in orders[i]:
        #             s_count=s_count+orders[i][j]
        #             if s_count<=S:
        #                 d_f=d_f+orders[i][j]*distance[0][location[j]]
        #                 del orders[i][j]
        #             if s_count>=S:
        #                 break
        #     if (s_count>=S) :        
        #         break   
        
        # ratio[s-1].append((d_f/S)/(d_0/5500.0))
        
        # #这是一段附加代码，如果要删除一部分订单的话需要添加     
        # s_count=0
        # seq=[i for i in range(order_n)]
        # random.shuffle(seq)
        # for i in seq:
        #     for j in orders[i].keys():           
        #         s_count=s_count+orders[i][j]
        #     if s_count<=S:
        #         orders[i]={}        
        #     if (s_count>=S):
        #         break
        
        
        #这是一段附加代码，如果要删除一部分SKU    
        s_count=0
        while 1:
            j=random.randint(1, goods+1)
            for i in range(order_n):    
                if j in orders[i]:
                    s_count=s_count+orders[i][j]
                    if s_count<=S:
                        del orders[i][j]
                    if s_count>=S:
                        break
            if (s_count>=S) :        
                break    
        
        
        #计算距离
        order=[{} for i in range(order_n)]  #以储位记录的订单,计算距离时只需记录访问节点，无需明确商品数量
        route=[0 for i in range(order_n)]    
        for i in range(order_n):
            temp1=[location[k] for k in orders[i]]
            for item in temp1:
                order[i][item] = temp1.count(item)
                
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
                
        # # 初始距离代码        
        # print(sum(route))      
        # origin.append(sum(route))
        # np.save("origin2.npy",origin)
        
        #PDR代码
        pdr=1-sum(route)/origin[s-1] 
        print(pdr)
        pdr_list[s-1].append(pdr)  
pdr_list=np.array(pdr_list)  
pdr_mean = np.mean(pdr_list,axis=0)
        
np.save("a",pdr_list)
# ratio= np.array(ratio)
# ratio_mean = np.mean(ratio,axis=0)