import numpy as np
from scipy.interpolate import lagrange
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl

def f(x):
    return (x/9.0)*np.exp(x*x/(-1*18))

#用拉格朗日法求出拟合函数
x = [1,4,7,10]
# y = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
y = [0.1051,0.1827,0.0511,0.0043]

P = lagrange(x, y)
print(P)

def g(x):
    return f(x)-P(x)

num = 9999
x = np.linspace(0,10, num)
err = np.zeros((num), dtype=float)
j = 0
for i in x:
    err[j] = g(i)
    j += 1

#得到绝对误差的最大值最小值
(min_err, max_err) = (min(err), max(err))
print((min_err, max_err))


fig = plt.figure(figsize=(10, 5))  
ax = fig.add_subplot(1, 1, 1)  
# ax.plot(x, g(x), lw=6)  
ax.plot(x, f(x),x,P(x), lw=1)
ax.set_ylabel(r'$error$', fontsize=18)
# ax.axhline(min_err, ls=':', color='k', lw=2)
# ax.axhline(max_err, ls=':', color='k', lw=2)
plt.show()
