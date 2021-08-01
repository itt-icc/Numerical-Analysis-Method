import sympy
import numpy as np
from scipy.interpolate import KroghInterpolator

def f(x):
    return (x/9.0)*np.exp(x*x/(-1*18))

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
    x=sympy.Symbol('x')#创建符号变量
    result=0.0
    for i in range(len(x_)):
        pass
        result+=y_[i]*get_L(x_[i],x_)(x)
    return result

def Netwon(X,Y):
    n=len(X)
    #先将表格制作出来
    A=np.zeros([n,n])
    for i in range(0,n):
        A[i][0] = Y[i]
    for j in range(1,n):
        for i in range(j,n):
            A[i][j] = (A[i][j-1] - A[i-1][j-1])/(X[i]-X[i-j])
    #创建函数表达式
    P=Y[0]
    x=sympy.Symbol('x')#创建符号变量
    for i in range(1,n):
        w=1
        for j in range(0,i):
            w*=(x-X[j])
        P+=w*A[i][i]
    return P

def Cubic(x_given,y_given):#直接按照书本上伪代码写
    n=len(x_given)
    a=np.zeros(n,dtype=float)
    b=np.zeros(n,dtype=float)
    c=np.zeros(n,dtype=float)
    d=np.zeros(n,dtype=float)
    s=np.zeros(n-1,dtype=tuple)#分段函数
    h=np.zeros(n-1,dtype=float)
    m=np.zeros(n-1,dtype=float)#α
    m[0]=0
    x=sympy.symbols('x')
    for i in range(0,n):
        a[i]=y_given[i]
    for i in range(0,n-1):
        h[i]=x_given[i+1]-x_given[i]
    for i in range(1,n-1):
        m[i]=3*(a[i+1]-a[i])/h[i]-3*(a[i]-a[i-1])/h[i-1]
    l=np.zeros(n,dtype=float)
    u=np.zeros(n,dtype=float)
    z=np.zeros(n,dtype=float)
    l[0],u[0],z[0]=1,0,0    
    for i in range(1,n-1):
        l[i]=2*(x_given[i+1]-x_given[i-1])-h[i-1]*u[i-1]
        u[i]=h[i]/l[i]
        z[i]=(m[i]-h[i-1]*z[i-1])/l[i]
    l[n-1],z[n-1],c[n-1]=1,0,0
    for j in range(n-2,-1,-1):
        c[j]=z[j]-u[j]*c[j+1]
        b[j]=(a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j]=(c[j+1]-c[j])/(3*h[j])
    for i in range(0,n-1):
        s[i]=a[i]+b[i]*(x-x_given[i])+c[i]*(x-x_given[i])**2+d[i]*(x-x_given[i])**3
    return s#分段函数
x = [1,4,7,10]
y = [0.1051,0.1827,0.0511,0.0043]

L=get_Lagrange(x,y)#拉格朗日插值法
N=Netwon(x,y)#牛顿插值法
D=Cubic(x,y)#样条插值法
H=KroghInterpolator(x, y)#埃米尔特插值
print('拉格朗日插值法:')
print(sympy.simplify(L))
print('牛顿插值法:')
print(sympy.simplify(N))
print('三次样条插值法:')
for i in range(len(x)-1):
    print('S('+str(i)+') = '+str(sympy.simplify(D[i])))