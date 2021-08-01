# hm2-problem-4-b
import math
import numpy as np  
import matplotlib.pyplot as plt

def Function(x,num):  #方程组
    i = num
    f = np.zeros((i),dtype=float) #在使用矩阵之前先创建矩阵
    # problem-4-b:
    f[0]=10*x[0]-2*x[1]**2+x[1]-2*x[2]-5
    f[1]=8*x[1]**2+4*x[2]**2-9
    f[2]=8*x[1]*x[2]+4
    return f

def Gredient(x,num):
    a=0.1**3#梯度步长
    x1 = np.copy(x)
    i = 0
    while( g(x,num)[0] < 0.001): 
        x1 = x-a*dg(x,num)
        x = x1
        i = i+1
    print(x)
    return x

def dg(x,num):
    dg = np.zeros((3),dtype=float) #在使用矩阵之前先创建矩阵
    dx=0.1**5
    for i in range(0,num):
        x1 = np.copy(x)
        x1[i]=x1[i]+dx
        dg[i]=(g(x1,num)[i]-g(x,num)[i])/dx
    return dg

def g(x,num):
    f=Function(x,num)
    g = np.zeros((3),dtype=float) #在使用矩阵之前先创建矩阵
    g[0]=f[0]**2+f[1]**2+f[2]**2
    g[1]=f[0]**2+f[1]**2+f[2]**2
    g[2]=f[0]**2+f[1]**2+f[2]**2
    return g

def invJacobian(x,num):                         #计算雅可比矩阵的逆矩
    df = np.zeros((num,num),dtype=float)
    dx = 0.1**5    
    for i in range(0,num):              # 求导数，i是列，j是行
        for j in range(0,num):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx   
            df[i,j] = (Function(x1,num)[i]-Function(x,num)[i])/dx  
    df_1 = np.linalg.inv(df)      #计算逆矩
    return df_1

def Newton(x,num):
    x1 = np.copy(x)
    i = 0
    delta = np.copy(x)
    while( np.sum(abs(delta)) > 0.05 and i < 20):  #控制循环次数
        x1 = x-np.dot(invJacobian(x,num),Function(x,num))  #公式子
        delta = x1-x #相对误差
        x = x1
        i = i+1
        print(x)
    print(i)
    return x

num =3     # 方程未知数的个数
x = np.ones((num),dtype=float)#初始
x1=Gredient(x,num)
a = Newton(x1,num)
print(a)
