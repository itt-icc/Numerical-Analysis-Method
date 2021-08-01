import numpy as np
from scipy.interpolate import lagrange
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl


def f(x):
    return np.sin(np.log(x))


#用拉格朗日法求出拟合函数
x = [2.0, 2.4, 2.6]
y = [f(x[0]), f(x[1]), f(x[2])]
P = lagrange(x, y)
print(P)

def g(x):
    return f(x)-P(x)

num = 9999
x = np.linspace(0, 0.6, num)
err = np.zeros((num), dtype=float)
err=g(x)

#得到绝对误差的最大值最小值
(min_err, max_err) = (min(err), max(err))
print((min_err, max_err))


#画出误差图像
fig = plt.figure(figsize=(10, 5))  
ax = fig.add_subplot(1, 1, 1)  
ax.plot(x, g(x), lw=6)  
ax.set_ylabel(r'$error$', fontsize=18)  
ax.axhline(min_err, ls=':', color='k', lw=2)
ax.axhline(max_err, ls=':', color='k', lw=2)
plt.show()
