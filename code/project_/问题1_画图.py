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
def get_L(xk,x_=[]):#计算Lnk(x)=(x-x0)...(x-xn)/(xk-x0)...(xk-xn)
    def L_k(x):
        numerator=1.0
        dominator=1.0
        for i in range(len(x_)):
            if x_[i]!=xk:
                dominator*=xk-x_[i]
                numerator*=x-x_[i]
        return numerator/dominator
    return L_k
def get_Lagrange(x_=[],y_=[]):#获得拉格朗日插值函数
    def Lagrange(x):
        result=0.0
        for i in range(len(x_)):
            pass
            result+=y_[i]*get_L(x_[i],x_)(x)
        return result
    return Lagrange

'''
三次样条插值
'''
from scipy import interpolate #导入scipy里interpolate模块中的interpld插值模块


'''
埃米尔特插值
'''
from scipy.interpolate import KroghInterpolator


'''
#牛顿插值法
'''
def get_diff_table(X,Y):
    n=len(X)
    A=np.zeros([n,n])
    for i in range(0,n):
        A[i][0] = Y[i]
    for j in range(1,n):
        for i in range(j,n):
            A[i][j] = (A[i][j-1] - A[i-1][j-1])/(X[i]-X[i-j])
    return A
def get_Newton_inter(X,Y):
    def netwon(x):
        sum=Y[0]
        temp=np.zeros((len(X),len(X)))
        #将第一行赋值
        for i in range(0,len(X)):
            temp[i,0]=Y[i]
        temp_sum=1.0
        for i in range(1,len(X)):
            #x的多项式
            temp_sum=temp_sum*(x-X[i-1])
            #计算均差
            for j in range(i,len(X)):
                temp[j,i]=(temp[j,i-1]-temp[j-1,i-1])/(X[j]-X[j-i])
            sum+=temp_sum*temp[i,i] 
        return sum
    return netwon


#数据输入部分
x = [1,4,7,10]
y = [0.1051,0.1827,0.0511,0.0043]
fy = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]


L=get_Lagrange(x,y)#拉格朗日插值法
N=get_Newton_inter(x,y)#牛顿插值法
I= interpolate.interp1d(x, y,kind="cubic")#3次样条插值
H = KroghInterpolator(x, y)#埃米尔特插值



#画图部分
num = 1000
x_cubic=np.linspace(np.array(x).min(),np.array(x).max(), num)#因为样条插值只能在给定区间中
x = np.linspace(0,10, num)

fig = plt.figure(figsize=(12, 7))  
ax = fig.add_subplot(1, 1, 1)  

ax.plot(x, f(x),'b',label='f(x)')
ax.plot(x, H(x),'--',label='Hermite')
ax.plot(x, N(x),'g',label='Newton')
ax.plot(x, L(x),'r--',label='Lagrange')
ax.plot(x_cubic, I(x_cubic),'b--',label='cubic',lw=0.5)

plt.sca(ax)
plt.plot([1,4,7,10], y, 'ro',label='given points')
plt.legend(loc=1)
ax.set_ylabel(r'$Approximation$', fontsize=18)
plt.show()
