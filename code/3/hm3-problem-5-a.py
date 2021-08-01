# hm3-problem-5-a

import numpy as np
import math

A = np.array([[3,-1,1], [3,6,2], [3,3,7]], dtype=float)
b = np.array([[1], [0], [4]], dtype=float)
Z = np.hstack((A, b))  # 将两个矩阵合成增广矩阵

n = 3  # 系数矩阵的阶数

D = np.zeros((n, n), dtype=float)
L = np.zeros((n, n), dtype=float)
U = np.zeros((n, n), dtype=float)

for i in range(0, n):
    D[i][i] = A[i][i]
print('D\n', D)
for i in range(1, n):
    for j in range(0, i):
        L[i][j] = A[i][j]*-1
print('L\n', L)
for i in range(0, n):
    for j in range(i+1, n):
        U[i][j] = A[i][j]*-1
print('U\n', U)
print('solution')

def Jacobi(D,L,U,b):
    TOL=1 
    X = np.zeros((n,1), dtype=float)
    X1 = np.zeros((n,1), dtype=float)
    #Jacobi method
    while TOL > 0.001:
        DLU = np.dot(np.linalg.inv(D), (L+U))
        X1 = np.dot(DLU, X)+np.dot(np.linalg.inv(D), b)
        TOL=max(np.abs(X-X1))#求无穷范数
        X = X1
        # print(X1)
    return X1
def Gauss_Seidel(D,L,U,b):
    TOL=1 
    X = np.zeros((n,1), dtype=float)
    X1 = np.zeros((n,1), dtype=float)
    #Jacobi method
    while TOL > 0.001:
        DLU = np.dot(np.linalg.inv(D-L),U)
        X1 = np.dot(DLU, X)+np.dot(np.linalg.inv(D-L), b)
        TOL=max(np.abs(X-X1))#求无穷范数
        X = X1
        # print(X1)
    return X1
X1=Jacobi(D,L,U,b)
print('Jacobi method\n',np.transpose(X1))
X2=Gauss_Seidel(D,L,U,b)
print('Gauss_Seidel method\n',np.transpose(X2))