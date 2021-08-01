# coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
 
x = np.ones((5),dtype=float)#初始值
print(x)
x1 = np.copy(x)
print(x1)
def Fun(x,num):  #方程组在这里，三个变量分别是x的三个分量，num是未知数个数，这里是3，f是三个方程组 
    i = num
    f = np.zeros((i),dtype=float) #在使用矩阵之前先创建矩阵
    f[0] = x[0]*x[1]-x[2]*x[2]-1.
    f[1] = x[0]*x[1]*x[2]+x[1]*x[1]-x[0]*x[0]-2.
    f[2] = math.exp(x[0])+x[2]-math.exp(x[1])-3.
    return f
print(Fun(x,3))