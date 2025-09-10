# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:02:15 2025

@author: 18272022928
"""

import pandas as pd
import random
import numpy as np
import math
# distance_matrix=[[0,1,1.4,1],[1,0,1,1.4],[1.4,1,0,1],[1,1.4,1,0]]
# from tsp import solve_tsp
from pytsp.christofides_tsp import christofides_tsp
from collections import defaultdict
from collections import OrderedDict
def merge_dicts_with_sum(dicts):
    merged = defaultdict(int)  # 使用int初始化，这样可以直接进行加法
    for d in dicts:
        for key, value in d.items():
            merged[key] += value
    return dict(merged)  # 返回普通字典
 



seeds=[i for i in range(1,11)]  #种子


# #初始距离代码
# S_lst=[0]
# origin=[]

#求PDR代码
S_lst=[250,500,1000,1500,2000]  #前置仓容量
# 初始距离,不同布局对应的距离不同
# origin = np.load('origin.npy', allow_pickle=True).tolist()  # 初始距离
# origin = np.load('origin1.npy', allow_pickle=True).tolist()  # 初始距离
origin = np.load('origin.npy', allow_pickle=True).tolist()  # 初始距离random
# S_lst=[250]
# seeds=[1]

l0=[[] for i in range(10)]
pdr_list=[[] for i in range(10)]
aisle_list=[[] for i in range(10)]
prop_list=[[] for i in range(10)]
size_list=[[] for i in range(10)]
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

nonstop=[i for i in range(16,31)]
#每个location隶属的通道
aisle=[]
for i in range (60):
    for k in range(25):
        aisle.append(i+1)




#train
random.seed(11)
days_train=10

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
        random.shuffle(location_sec)
        
        location_t.append(location_sec)               

# 移除并保存元素
element = location_t.pop(16-1)
# 在新位置插入元素
location_t.insert(8, element)

for i in location_t:
    location.extend(i)
location.insert(0, 0)

np.save("location.npy",location)



# 订单的生成
order_n =1000
goods = len(location)-1
    
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
  
#生成累加权重
    for i in range(good_sec[k]):
        weight_sec[i]=sales[k]* weight_sec[i]
        if k>=1:
            weight_sec[i]=past+ weight_sec[i]
    weight.extend(weight_sec)      
    

order_train=[[{} for i in range(order_n)] for j in range(days_train)]
for d in range(days_train):
    order_num=[0 for i in range(order_n)]
    for i in range(order_n):
        order_num[i]=random.randint(1,10)
    
    for i in range(order_n):
        temp = random.choices(choices, cum_weights=weight, k=order_num[i]) #累加权重，求解速度快 
        for item in temp:
            order_train[d][i][item] = temp.count(item)       


#统计各商品的平均需求
result_list=[]
for d in  range(days_train):          
    result_list.append(merge_dicts_with_sum(order_train[d]))
result=merge_dicts_with_sum(result_list)
demand={}
for j in result:
    demand[j]=result[j]/(days_train) 
for j in range(1,goods+1):   
    if j not in demand:
        demand[j]=0
#统计各商品出现的频次

freq=[[0 for _ in range(days_train)] for _ in range(goods)]

for d in  range(days_train):     
    for i in range(order_n):
        for j in order_train[d][i]:
            freq[j-1][d]+=1                
freqency=np.mean(freq,axis=1)

        

for S in S_lst:
    # iteration=0
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
        
        #随机location需要加载原布局
        location = np.load('location.npy', allow_pickle=True).tolist() 
        
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
        aisle_num=0
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

            #统计每个订单的原始通道个数    
            temp3 = [aisle[location[k]-1] for k in temp]
            temp4=list(set(temp3))
            aisle_num+=len(temp4)
            if temp3[0]  not in nonstop:
                aisle_num+=1
            if temp3[-1]  not in nonstop:
                aisle_num+=1

        # # 计算距离
        # d_0=0
        # for i in range(order_n):
        #     for j in orders[i]:
        #         d_0+=distance[0][location[j]]*orders[i][j]    
        
        
        prop=[]        
        size=[]
        
            
        # #这是一段附加代码，要对流行商品进行删除，静态        
        # sorted_items = sorted(result.items(), key=lambda x: x[1],reverse=True) #排序
        # sku_p=[item[0] for item in sorted_items]
      
        # temp=0
        # for j in sku_p:
        #     popular=0   
        #     for i in range(order_n): 
        #         if j in orders[i]:
        #             popular=popular+orders[i][j]                    
        #             if popular<=demand[j]:
        #                 for count in range(orders[i][j]):
        #                     prop.append(j)
        #                     size.append(order_num[i])
        #                 del orders[i][j]

        #             if popular>=demand[j]:
        #                 break
        #     temp=temp+demand[j]         
        #     if (temp>=S) :        
        #         break 

        
         
        
        # # 这是一段附加代码，要对远距离商品进行删除 ,静态
        # d_f=0
        # f=[i for i in distance[0]]
        # s_count=0
        # sku_p=sorted(range(len(location)),key=lambda k: f[location[k]],reverse=True) 
        # for j in sku_p:
        #     popular=0
        #     # if j not in demand:
        #     #     continue
        #     for i in range(order_n):    
        #         if j in orders[i]:
        #             popular=popular+orders[i][j]
        #             if popular<=demand[j]:
        #                 d_f=d_f+orders[i][j]*distance[0][location[j]]
        #                 for count in range(orders[i][j]):
        #                     prop.append(j)
        #                 del orders[i][j]
        #             if popular>=demand[j]:
        #                 break
        #     s_count=s_count+demand[j]            
        #     if (s_count>=S) :        
        #         break   
        
        # ratio[s-1].append((d_f/S)/(d_0/5500.0))
        
        
        
        # #这是一段附加代码，如果要删除一部分SKU,静态 
        # random.seed(1)                                                                                                                                                                                                                     
        # rand_seq=[i  for i in range(1, goods+1)]
        # random.shuffle(rand_seq)
        
        # s_count=0
       
        # for j in rand_seq  :
        #     # j=random.randint(1, goods+1)
            
        #     if j not in demand:
        #         continue           
        #     popular=0
        #     for i in range(order_n):    
        #         if j in orders[i]:
        #             popular=popular+orders[i][j]
        #             if popular<=demand[j]:
        #                 for count in range(orders[i][j]):
        #                     prop.append(j)   
        #                     size.append(order_num[i])
        #                 del orders[i][j]              
        #             if popular>=demand[j]:
        #                 break
        #     s_count=s_count+demand[j]          
        #     if (s_count>=S) :        
        #         break    
            

        # #bang-for-buck
        
        #首先计算每个SKU对应的排序p/sqrt(f) p是需求频次,f需求量/体积
        rank=[0 for j in range(goods)]
        for j in range(goods):
            if freqency[j]==0:
                rank[j]=0
            else:
                rank[j]=freqency[j]/math.sqrt(demand[j+1])
        sku_p=sorted(range(len(rank)),key=lambda k: rank[k],reverse=True) 
        # sku_p=[j+1 for j in sku_p]
        
        #计算相关参数       
        cr=sum( distance[0][location[i+1]] for i in range(goods))/goods
        #体积归一化
        inventory={}
        for key in demand:
            inventory[key]=demand[key]/S
        #贪婪算法
        f_=0
        for k in range(1,goods+1):       #选前k个商品
            #计算前k个商品的目标值
            f=sum(cr*freqency[sku_p[j]]-cr*math.sqrt(inventory[sku_p[j]+1])*sum(math.sqrt(inventory[sku_p[i]+1]) for i in range(k)) for j in range(k))
            #与前一个值比较,若小于,则前一个值为最大值，结束循环；若大于，保存下来，继续循环  
           
            if f<f_:
                k_min=k-1
                break
            elif f==f_:
                k_min=k
                break
            else:
                f_=f
            
        #计算前k个商品各自的体积    
        Q={}    
        for j in range(k_min):      
            Q[sku_p[j]+1]=S*math.sqrt(inventory[sku_p[j]+1])/sum(math.sqrt(inventory[sku_p[i]+1]) for i in range(k_min))            
        
        #消耗掉库存Q     
        for j in Q :
            popular=0
            for i in range(order_n):
                if j in orders[i]:
                    popular=popular+orders[i][j]
                    if popular<=Q[j]:
                        del orders[i][j]
                    if popular>=Q[j]:
                        break        
            
            
        # # 其余指标
        # prop1=[]
        # for item in prop:
        #     prop1.append(location[item])
        # prop1=list(dict.fromkeys(prop1))                  
        # prop=list(dict.fromkeys(prop))
        # prop_list[s-1].append(len(prop))        
        # size=np.mean(size)
        # size_list[s-1].append(size)


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
                
        # # 初始距离代码        
        # print(sum(route))      
        # origin.append(sum(route))
        # np.save("origin.npy",origin)
        
        #PDR代码
        pdr=1-sum(route)/origin[s-1] 
        print(pdr)
        pdr_list[s-1].append(pdr)  
        aisle_list[s-1].append((aisle_num-aisle_num_1)/aisle_num)
        # iteration+=1
pdr_list=np.array(pdr_list)  
pdr_mean = np.mean(pdr_list,axis=0)

aisle_list=np.array(aisle_list)
aisle_mean=np.mean(aisle_list,axis=0)        
np.save("b",pdr_list)



# prop_list=np.array(prop_list)
# prop_mean=np.mean(prop_list,axis=0)
# size_mean = np.mean(size_list,axis=0)

# ratio= np.array(ratio)
# ratio_mean = np.mean(ratio,axis=0)