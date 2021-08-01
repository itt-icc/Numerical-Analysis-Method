def Natural(x_given,y_given,n):
    a=np.zeros(n,dtype=float)
    b=np.zeros(n,dtype=float)
    c=np.zeros(n,dtype=float)
    d=np.zeros(n,dtype=float)
    s=np.zeros(n-1,dtype=tuple)#数据类型居然是元组！！
    h=np.zeros(n-1,dtype=float)
    m=np.zeros(n-1,dtype=float)#相当于α
    m[0]=0
    #x=sympy.symbols('x')
    for i in range(0,n):
        a[i]=y_given[i]
    for i in range(0,n-1):
        h[i]=x_given[i+1]-x_given[i]
    for i in range(1,n-1):
        m[i]=3*(a[i+1]-a[i])/h[i]-3*(a[i]-a[i-1])/h[i-1]
    l=np.zeros(n,dtype=float)
    u=np.zeros(n,dtype=float)
    z=np.zeros(n,dtype=float)
    l[0]=1
    u[0]=0
    z[0]=0
    for i in range(1,n-1):
        l[i]=2*(x_given[i+1]-x_given[i-1])-h[i-1]*u[i-1]
        u[i]=h[i]/l[i]
        z[i]=(m[i]-h[i-1]*z[i-1])/l[i]
    l[n-1]=1
    z[n-1]=0
    c[n-1]=0
    for j in range(n-2,-1,-1):
        c[j]=z[j]-u[j]*c[j+1]
        b[j]=(a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j]=(c[j+1]-c[j])/(3*h[j])
    for i in range(0,n-1):
        s[i]=a[i]+b[i]*(x-x_given[i])+c[i]*(x-x_given[i])**2+d[i]*(x-x_given[i])**3
    return s
