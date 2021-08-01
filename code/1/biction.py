
'''
Bisection法求解方程的根
'''
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import numpy as np

# f = lambda x :np.exp(x)-x**2+3*x-2
# tol = 0.00001
# a,b = 0,1
f = lambda x :np.cos(x)-2*x*x+3*x-1
tol = 0.00001
a,b = 1.2,1.3


x = np.linspace(a-1,b+1,1000)

#fig,ax = plt.subplots(1,1,figsize = (10,4))#同时定义子图的个数，以及画幅大小#日后数据可视化很重要
fig=plt.figure(figsize=(10,5))#创建一个画布
ax= fig.add_subplot(1,1,1)#在画布中定义一个坐标轴图像将来画出对应的值
ax.plot(x,f(x),lw = 0.5)#指定线的宽度
ax.axhline(0,ls=':',color='k')#水平参考线(y,c,ls,lw)y:出发点，c:线的颜色，ls:线条的风格，lw:线的宽度
ax.set_xticks(list(np.arange(a-1,b+2,1)))#设置刻度值即坐标轴刻度值
ax.set_xlabel(r'$x-axis$',fontsize=10)#x轴坐标名称设置
ax.set_ylabel(r'$f(x)$',fontsize=18)#Y轴坐标名称设置
fa,fb = f(a),f(b)
ax.plot(a,fa,'o')#plot(x,y2,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
ax.plot(b,fb,'o')
ax.text(a,fa+0.5,r'$a$',ha='center',fontsize=18)#主要是添加文本信息%%可以很秀
ax.text(b,fb+0.5,r'$b$',ha='center',fontsize=18)

n = 1
midpoint=[]
while b - a > tol:
    m = a + (b - a) / 2
    midpoint.append(m)
    fm = f(m)
    ax.plot(m,fm,'o')
    # ax.text(m,fm - 3,r'$m_%d$' %n,ha = 'center')
    n += 1
    if np.sign(fa) == np.sign(fm):
        a,fa = m,fm
    else:
        b,fb = m,fm
print(n)
print(midpoint)
ax.plot(m,fm,'r*',markersize=10)
ax.annotate("Root approximately at %.10f" % m,
            fontsize=14,family='serif',
            xy = (a,fm),xycoords='data',
            xytext=(-150,-50),textcoords='offset points',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-0.5'))##适合采用箭头进行标注

ax.set_title('Bisection method')
plt.show()