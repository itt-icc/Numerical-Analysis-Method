# problem3 b:
import math
import numpy as np  
import matplotlib.pyplot as plt

def Function(x,num):  #方程组
    i = num
    f = np.zeros((i),dtype=float) #在使用矩阵之前先创建矩阵
    f[0]=x[0]**2+x[1]-37.0
    f[1]=x[0]-x[1]**2-5.0
    f[2]=x[1]+x[2]+x[0]-3
    return f

def invJacobian(x,num):   #计算雅可比矩阵的逆矩阵
    df = np.zeros((num,num),dtype=float)
    dx = 0.1**10 #设置求导精度
    for i in range(0,num):   # 求导数，i是列，j是行
        for j in range(0,num):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx  
            df[i,j] = (Function(x1,num)[i]-Function(x,num)[i])/dx   
    df_1 = np.linalg.inv(df)    #计算逆矩阵
    return df_1

def Newton(x,num):
    x1 = np.copy(x)
    i = 0
    while( i < 15):  #控制循环次数
        x1 = x-np.dot(invJacobian(x,num),Function(x,num))  #公式
        x = x1
        i = i+1
        print(x)
    print(i)
    return x

num =3      # 方程组阶数
x = np.zeros((num),dtype=float)#初始值
print(x)
a = Newton(x,num)
print(a)
