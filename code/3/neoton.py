from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import numpy as np
import sympy

tol = 10**(-8)
xk = 0

s_x = sympy.symbols('x')# 首先定义x为一个符号，表示一个有符号的变量
s_f = -1*sympy.cos(s_x) - s_x**3

f = lambda x : sympy.lambdify(s_x,s_f,'numpy')(x)
fp = lambda x : sympy.lambdify(s_x,sympy.diff(s_f,s_x),'numpy')(x)
# 可视化方程求解过程
x = np.linspace(-2.1,2.1,1000)
fig=plt.figure(figsize=(12,5))#创建一个画布
ax= fig.add_subplot(1,1,1)
ax.plot(x,f(x))
ax.axhline(0,ls=':',color='k')
n = 0
xk_k=[]
while f(xk) > tol or n<2:
    xk_k.append(xk)
    xk_new = xk - f(xk) / fp(xk)
    
    ax.plot([xk, xk], [0, f(xk)], color='k', ls=':')#点对点划线
    ax.plot(xk,f(xk),'ko')
    ax.text(xk,-0.5,r'$x_%d$' % n,ha='center')
    ax.plot([xk,xk_new],[f(xk),0],'k-')

    xk = xk_new
    n += 1
print(n)
print(xk_k)
ax.plot(xk,f(xk),'r*',markersize=15)
ax.annotate("Root approximately at %.9f" % xk,
            fontsize=14,family='serif',
            xy = (xk,f(xk)),xycoords='data',
            xytext=(-150,+50),textcoords='offset points',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-0.5'))

ax.set_title('Newton method')
ax.set_xticks([-1,0,1,2])
plt.show()