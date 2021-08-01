import numpy as np
from matplotlib import pyplot as plt


def f(t, w):#函数
    return 1+w/t+(w/t)*(w/t)

h=0.2
a,b = 1, 3
n = (b-a)/h
t = a
w = 0
Y = []
for i in range(1,int(n)+2):#欧拉法
    Y.append(w)
    w += h*f(t, w)
    t = a+i*h

fig = plt.figure(figsize=(8, 5))  
ax = fig.add_subplot(1, 1, 1)  
x = np.linspace(a,b, n+1)
ax.plot(x, Y,'*',label='euler')
plt.legend(loc=4)
plt.show()
print(x)
print(Y)