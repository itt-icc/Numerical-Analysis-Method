import numpy as np
from scipy.interpolate import lagrange
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
from scipy.interpolate import KroghInterpolator

def f(x):#原函数
    return (x/9.0)*np.exp(x*x/(-1*18))

#牛顿插值法
def get_order_diff_quot(xi=[], fi=[]):#计算n阶差商 f[x0, x1, x2 ... xn]
    if len(xi) > 2 and len(fi) > 2:
        return (get_order_diff_quot(xi[:len(xi) - 1], fi[:len(fi) - 1]) - get_order_diff_quot(xi[1:len(xi)], fi[1:len(fi)])) / float(xi[0] - xi[-1])
    return (fi[0] - fi[1]) / float(xi[0] - xi[1])
def get_Wi(i=0, xi=[]):#计算(x - x0)(x - x1)(x - x2)...(x-xn)
    def Wi(x):
        result = 1.0
        for each in range(i):
            result *= (x - xi[each])
        return result
    return Wi
def get_Newton_inter(xi=[], fi=[]):#获得牛顿插值函数
    def Newton_inter(x):
        result = fi[0]
        for i in range(2, len(xi)):
            result += (get_order_diff_quot(xi[:i],fi[:i]) * get_Wi(i-1, xi)(x))
        return result
    return Newton_inter



#数据输入部分
x = [1,4,7,10]
y = [0.1051,0.1827,0.0511,0.0043]
fy = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]

L = lagrange(x, y)#lagrange

H = KroghInterpolator(x, y)#hermite

N=get_Newton_inter(x, y) 



num = 9999
x = np.linspace(0,10, num)
err_L = np.zeros((num), dtype=float)
err_H = np.zeros((num), dtype=float)

j = 0
for i in x:
    err_L[j] = f(i)-L(i)
    err_H[j] = f(i)-(i)
    j += 1

#得到绝对误差的最大值最小值
(min_err_L, max_err_L) = (min(err_L), max(err_L))
print('lagrange\n'+str((min_err_L, max_err_L)))


fig = plt.figure(figsize=(12, 7))  
ax = fig.add_subplot(1, 1, 1)  
# ax.plot(x, g(x), lw=6)  
# ax.plot(x, f(x),label='f')
# ax.plot(x,L(x),'ro',label='lagrange',lw=0.1)
# ax.plot(x,H(x),'g--',label='Hermite Interpolation')
ax.plot(x, f(x)-L(x),label='f')
plt.legend(loc=1)
ax.set_ylabel(r'$error$', fontsize=18)
# ax.axhline(min_err, ls=':', color='k', lw=2)
# ax.axhline(max_err, ls=':', color='k', lw=2)
plt.show()
