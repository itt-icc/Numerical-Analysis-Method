# hm3-problem-4-a

import numpy as np
import math

A = np.array([[4, 1, -1], [-1, 3, 1], [2, 2, 5]], dtype=float)
b = np.array([[5], [-4], [1]], dtype=float)
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



interation = 0
X = np.zeros((n,1), dtype=float)
X1 = np.zeros((n,1), dtype=float)

while interation <3:
    interation += 1
    X1=np.dot(np.linalg.inv(D),np.dot(L+U,X)+b)
    print('solution'+' '+str(interation)+':')
    X = X1
    print(np.transpose(X1))
    # print('error')
    # print(b-np.dot(A,X1))