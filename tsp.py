# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:52:47 2024

@author: 18272022928
"""

import numpy as np
import math

# 定义城市个数及其距离矩阵
# N = 4  # 假设有4个城市
# dist_matrix = np.array([
#     [0, 10, 15, 20],
#     [10, 0, 35, 25],
#     [15, 35, 0, 30],
#     [20, 25, 30, 0]
# ])

# 当前路径和最优路径
best_path = []
best_cost = float('inf')  # 初始最优值为无穷大

def branch(dist_matrix,path, visited):
    N=len(dist_matrix)
    # 获取当前城市
    current_city = path[-1]
    
    # 遍历所有城市
    for next_city in range(N):
        # 如果城市未被访问
        if not visited[next_city]:
            # 创建新的路径和访问状态
            new_path = path + [next_city]
            # print(new_path)
            new_visited = visited.copy()
            new_visited[next_city] = True
            
            # 递归调用
            bound(dist_matrix,new_path, new_visited)
            
def calculate_bound(dist_matrix,path, visited):
    total_cost = 0
    N=len(dist_matrix)
    # 计算路径的总成本
    for i in range(len(path) - 1):
        total_cost += dist_matrix[path[i], path[i + 1]]
    
    # # 计算启发式下界（即估算未访问城市的最小边）
    # min_cost = 0
    # for i in range(N):
    #     if not visited[i]:
    #         min_cost += np.min(dist_matrix[i][visited])
    
    # return total_cost + min_cost
    return total_cost            

def bound(dist_matrix,path, visited):
    global best_cost, best_path
    N=len(dist_matrix)
    if len(path) == N:  # 如果所有城市都已访问
        # 计算回到起始城市的成本
        total_cost = calculate_bound(dist_matrix,path + [path[0]], visited)
        if total_cost < best_cost:  # 更新最优值
            best_cost = total_cost
            best_path = path + [path[0]]  # 更新最优路径
    else:
        # 计算当前路径的下界
        # print(path)
        
        lower_bound = calculate_bound(dist_matrix,path, visited)
        # print(lower_bound)
        # print(visited)
        if lower_bound < best_cost:  # 仅在可能情况下递归
            branch(dist_matrix,path, visited)
            
def solve_tsp(dist_matrix):
    global best_cost, best_path
    best_path = []
    best_cost = float('inf')  # 初始最优值为无穷大    
    N=len(dist_matrix)
    # 初始化路径和访问状态
    initial_path = [0]  # 从城市0开始
    visited = [False] * N  # 所有城市未访问
    visited[0] = True  # 设定初始城市已访问
    
    branch(dist_matrix,initial_path, visited)  # 开始分支定界

    # 输出结果
    # print(f"最优路径: {best_path}, 最小成本: {best_cost}")
    # return best_cost
    return best_path
