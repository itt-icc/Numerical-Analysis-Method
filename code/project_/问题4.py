from scipy import interpolate  # 导入scipy里interpolate模块中的interpld插值模块
import numpy as np
from scipy.interpolate import lagrange
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl


'''
#原函数
'''
def f(x):
    return (x/9.0)*np.exp(x*x/(-1*18))


'''
#拉格朗日插值法
'''
def get_L(xk, x_=[]):  # 计算Lnk(x)=(x-x0)...(x-xn)/(xk-x0)...(xk-xn)
    def L_k(x):
        numerator = 1.0
        dominator = 1.0
        for i in range(len(x_)):
            if x_[i] != xk:
                dominator *= xk-x_[i]
                numerator *= x-x_[i]
        return numerator/dominator
    return L_k


def get_Lagrange(x_=[], y_=[]):  # 获得拉格朗日插值函数
    def Lagrange(x):
        result = 0.0
        for i in range(len(x_)):
            pass
            result += y_[i]*get_L(x_[i], x_)(x)
        return result
    return Lagrange

'''
三次样条插值
'''
from scipy import integrate

'''
埃米尔特插值
'''
from scipy.interpolate import KroghInterpolator


'''
#牛顿插值法
'''
def get_diff_table(X, Y):
    n = len(X)
    A = np.zeros([n, n])
    for i in range(0, n):
        A[i][0] = Y[i]
    for j in range(1, n):
        for i in range(j, n):
            A[i][j] = (A[i][j-1] - A[i-1][j-1])/(X[i]-X[i-j])
    return A
def get_Newton_inter(X, Y):
    def netwon(x):
        sum = Y[0]
        temp = np.zeros((len(X), len(X)))
        # 将第一行赋值
        for i in range(0, len(X)):
            temp[i, 0] = Y[i]
        temp_sum = 1.0
        for i in range(1, len(X)):
            # x的多项式
            temp_sum = temp_sum*(x-X[i-1])
            # 计算均差
            for j in range(i, len(X)):
                temp[j, i] = (temp[j, i-1]-temp[j-1, i-1])/(X[j]-X[j-i])
            sum += temp_sum*temp[i, i]
        return sum
    return netwon

# 数据输入部分
x=[0,1,2,3,4,5,6,7,8,9,10]
y=[]
x_p = x
for i in x:
    y.append(round(f(i),2))
for i in range(len(y)):
    pass
    # y[i]+=0.002
point=[]
for i in y:
    point.append(i)

    
L = get_Lagrange(x, y)  # 拉格朗日插值法
N = get_Newton_inter(x, y)  # 牛顿插值法
I = interpolate.interp1d(x, y, kind="cubic")  # 3次样条插值
H = KroghInterpolator(x, y)  # 埃米尔特插值


# 画图部分
num = 1000
x_cubic = np.linspace(np.array(x).min(), np.array(
    x).max(), num)  # 因为样条插值只能在给定区间中
x = np.linspace(0, 10, num)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(3, 1, 1)
ax.plot(x, f(x), 'b', label='f(x)')
ax.plot(x, H(x), '--', label='Hermite')
ax.plot(x, N(x), 'g', label='Newton')
ax.plot(x, L(x), 'r--', label='Lagrange')
ax.plot(x_cubic, I(x_cubic), 'b--', label='cubic', lw=0.5)
plt.sca(ax)
plt.plot(x_p, point, 'ro', label='given points')
plt.legend(loc=1)
ax.set_ylabel(r'$Approximation$', fontsize=18)

ax1 = fig.add_subplot(3,1, 2)
ax1.plot(x, H(x)-f(x),'--',label='Hermite')
ax1.plot(x, N(x)-f(x),'g',label='Newton')
ax1.plot(x, L(x)-f(x),'r--',label='Lagrange')
ax1.plot(x_cubic, I(x_cubic)-f(x_cubic),'b--',label='cubic',lw=0.5)
plt.legend(loc=1)
ax1.set_ylabel(r'$absolute error$', fontsize=18)

ax2 = fig.add_subplot(3,1, 3)
ax2.plot(x, (H(x)-f(x))**2,'--',label='Hermite')
ax2.plot(x, (N(x)-f(x))**2,'g',label='Newton')
ax2.plot(x, (L(x)-f(x))**2,'r--',label='Lagrange')
ax2.plot(x_cubic, (I(x_cubic)-f(x_cubic))**2,'b--',label='cubic',lw=0.5)
plt.legend(loc=1)
ax2.set_ylabel(r'$squre error$', fontsize=18)
plt.show()

'''
L2范数
'''
def H_f(x):
    return (H(x)-f(x))**2
def I_f(x):
    return (I(x)-f(x))**2
def N_f(x):
    return (N(x)-f(x))**2
def L_f(x):
    return (H(x)-f(x))**2

L_2_H = integrate.quad(H_f, 0, 10)
L_2_L = integrate.quad(L_f, 0, 10)
L_2_I = integrate.quad(I_f, np.array(x_p).min(), np.array(x_p).max())
L_2_N = integrate.quad(N_f, 0, 10)

print('Lagrange L_2:'+str(L_2_L))  # 拉格朗日L2范数
print('Netwon L_2:'+str(L_2_N))  # 牛顿法L2范数
print('Cubic L_2:'+str(L_2_I))  # 分段插值L2范数
print('Hermite L_2:'+str(L_2_H))  # 埃米尔特插值L2范数
