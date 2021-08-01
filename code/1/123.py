import math
import numpy as np
from matplotlib import pyplot as plt

def F(S):
    y=0.072
    t=10/360*2*math.pi
    R=0.125*10E-3
    #M=(4-math.pi)*R*R
    M=0.375*0.375*8*R**3/(3*(1-0.375))
    H=7*10E-4
    return (-4*y*math.cos(t)*math.sqrt(S*M*math.pi/R))/H
S=np.linspace(0,1,1000)
# plt.figure(figsize=(12,12))
# plt.figure(1)
# plt.plot(S,F(S),lw=0.5)

fig = plt.figure(figsize=(10, 5))  # 创建一个画布
ax = fig.add_subplot(1, 1, 1)  # 在画布中定义一个坐标轴图像将来画出对应的值
ax.plot(S, F(S), lw=0.5)  # 指定线的宽度
ax.plot(S, S, lw=0.5)  # 指定线的宽度
plt.show()
print(F(0.1))