# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:09:09 2024

@author: 18272022928
"""
import sys
import numpy as np
import copy
# 用于表示无穷大的值

def floyd_warshall(graph):
    dist = copy.deepcopy(graph)
    num_vertices = len(graph)
    
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

l1=4  #distance of adjacent aisle
l2=4  #width of cross-aisle
l3=1  #distance of adjacent storage 
l4=10  #distance of adjacent floor

# # small instance
# row=5
# column=4
# cross=2
# point=row*column+column*cross
# location=row*column
# INFINITY = sys.maxsize
# graph=[[INFINITY for col in range(point)] for row in range(point)]

# for i in range(point):
#     graph[i][i]=0
# for i in range(column):
#     for j in range(row-1):
#         graph[row*i+j][row*i+j+1]=l3
#         graph[row*i+j+1][row*i+j]=l3

# for i in range(cross):
#     for j in range(column-1):
#         graph[location+column*i+j][location+column*i+j+1]=l1
#         graph[location+column*i+j+1][location+column*i+j]=l1
        
# for i in range(cross):
#     for j in range(column):
#         graph[location+column*i+j][row*j+(row-1)*i]=(l2+l3)/2
#         graph[row*j+(row-1)*i][location+column*i+j]=(l2+l3)/2



# large instance
row=25
column=30
column_1=15 #第一层通道数
column_2=15  #第二层通道数
cross=3
point=row*column*2+column*cross
location=row*column*2  
INFINITY = sys.maxsize
graph=[[INFINITY for col in range(point)] for row in range(point)]

for i in range(point):
    graph[i][i]=0
for i in range(column*2):
    for j in range(row-1):
        graph[row*i+j][row*i+j+1]=l3
        graph[row*i+j+1][row*i+j]=l3


for i in range(cross):
    for j in range(column-1):
        graph[location+column*i+j][location+column*i+j+1]=l1
        graph[location+column*i+j+1][location+column*i+j]=l1

#两层的衔接处        
graph[location+column+column_1-1][location+column+column_1]=l4
graph[location+column+column_1][location+column+column_1-1]=l4

graph[location+column*0+column_1-1][location+column*0+column_1]=INFINITY
graph[location+column*0+column_1][location+column*0+column_1-1]=INFINITY 

graph[location+column*2+column_1-1][location+column*2+column_1]=INFINITY
graph[location+column*2+column_1][location+column*2+column_1-1]=INFINITY         
        

for i in range(2):
    for j in range(column):
        graph[location+column*i+j][row*j+(row-1)*i]=(l2+l3)/2
        graph[row*j++(row-1)*i][location+column*i+j]=(l2+l3)/2
        
        
for i in range(2):
    for j in range(column):
        graph[location+column*(i+1)+j][row*j+(row-1)*i+row*column]=(l2+l3)/2
        graph[row*j+(row-1)*i+row*column][location+column*(i+1)+j]=(l2+l3)/2
        

dist=floyd_warshall(graph) 
distance=[[0 for col in range(location+1)] for row in range(location+1)]
for i in range(location):
    for j in range(location):
        distance[i+1][j+1]=dist[i][j]
        
# #若前置仓在一层        
# for j in range(location): 
#     distance[0][j+1] = dist[location][j] 
#     distance[j+1][0] = dist[location][j] 

# np.save('distance.npy',distance)    

#若前置仓在二层    
for j in range(location): 
    distance[0][j+1] = dist[location+column_1][j] 
    distance[j+1][0] = dist[location+column_1][j]

# np.save('graph.npy',graph) 
np.save('distance.npy',distance) 
np.save('dist.npy',dist) 

