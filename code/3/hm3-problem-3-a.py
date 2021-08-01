# hm3-problem-3-a
import numpy as np
import math
A = np.array([[5.31, -6.10], [0.03, 58.9]], dtype=float)
b = np.array([[47.0], [59.2]], dtype=float)
Z = np.hstack((A, b))  # 将两个矩阵合成增广矩阵
flag = 0  # 有无唯一解标志位
n = 2  # 系数矩阵的阶数
j = 0
while j < n:
    for i in range(j, n):  # 替换两行减小误差，以及判断是否有唯一解
        if np.abs(Z[j][j]) < np.abs(Z[i][j]):
            tmp = np.copy(Z[j])
            Z[j] = Z[i]
            Z[i] = tmp
    j += 1
print('low error matrix:\n', Z)
if(Z[0][0] == 0):
    flag = 1
    print('no unique solution exist!')
for i in range(0, n-1):  # 高斯消元法
    if(flag == 1):
        break
    for j in range(i+1, n):
        m = Z[j][i]/Z[i][i]
        Z[j] -= m*Z[i]
if(Z[n-1][n-1] == 0):
    flag = 1
    print('no unique solution exist!')
X = np.zeros((n), dtype=float)
X[n-1] = Z[n-1][n]/Z[n-1][n-1]
print('gaussian elimination matrix:\n', Z)
if flag == 0:
    for i in range(n-2, -1, -1):  # 将结果输出
        Sum = 0.
        for j in range(i+1, n):
            Sum += Z[i][j]*X[j]
        X[i] = (Z[i][n]-Sum)/Z[i][i]
    print('solution:\n', X)
