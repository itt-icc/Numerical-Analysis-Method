# hm1-problem-6-a
import math
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def f(x):  # 定义函数
    return x+(x+2*np.sin(math.pi*x))/3
    
tol = 0.01
a, b = 1,2
x0 = 1


x = np.linspace(a-1, b+1, 1000)
# fig,ax = plt.subplots(1,1,figsize = (10,4))#同时定义子图的个数，以及画幅大小#日后数据可视化很重要
fig = plt.figure(figsize=(10, 5))  # 创建一个画布
ax = fig.add_subplot(1, 1, 1)  # 在画布中定义一个坐标轴图像将来画出对应的值
ax.plot(x, f(x), lw=0.5)  # 指定线的宽度
ax.plot(x, x, lw=0.5)  # 指定线的宽度
# 水平参考线(y,c,ls,lw)y:出发点，c:线的颜色，ls:线条的风格，lw:线的宽度
ax.axhline(0, ls=':', color='k')
ax.set_xticks(list(np.arange(a-1, b+2, 1)))  # 设置刻度值即坐标轴刻度值
ax.set_xlabel(r'$x-axis$', fontsize=10)  # x轴坐标名称设置
ax.set_ylabel(r'$f(x)$', fontsize=18)  # Y轴坐标名称设置
fa, fb = f(a), f(b)
# plot(x,y2,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
ax.plot(a, fa, 'o')
ax.plot(b, fb, 'o')
ax.text(a, fa+0.5, r'$a$', ha='center', fontsize=18)  # 主要是添加文本信息%%可以很秀
ax.text(b, fb+0.5, r'$b$', ha='center', fontsize=18)

n = 1
pn = []
x = f(x0)
while 1:
    x = f(x0)
    pn.append(x)
    if np.abs(x0-x) < tol:
        break
    ax.plot(x, f(x), 'o')
    n = n+1
    x0 = x
    pass
print(n)
print(pn)
ax.plot(x, x, 'r*', markersize=3)
ax.annotate("Root approximately at %.20f" % x,
            fontsize=14, family='serif',
            xy=(x, x), xycoords='data',
            xytext=(-150, -50), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5'))  # 适合采用箭头进行标注

ax.set_title('Fixed-point method')
plt.show()
