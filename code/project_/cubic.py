import numpy as np, matplotlib.pyplot as plt
from scipy import interpolate #导入scipy里interpolate模块中的interpld插值模块
x= np.array([0, 1, 2, 3, 4, 5, 6, 7])
y= np.array([3, 4, 3.5, 2, 1, 1.5, 1.25, 0.9]) #离散点的分布
xx = np.linspace(x.min(), x.max(), 100) #新的插值区间及其点的个数
plt.scatter(x, y) #散点图
#for n in ['linear','zero', 'slinear', 'quadratic', 'cubic', 4, 5]: #python scipy里面的各种插值函数
f = interpolate.interp1d(x, y,kind="cubic") #编辑插值函数格式
ynew=f(xx) #通过相应的插值函数求得新的函数点
plt.plot(xx,ynew,"g") #输出新的函数点的图像
plt.show()
