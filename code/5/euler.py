import numpy as np
from matplotlib import pyplot as plt


def f(t, w):
    pass
    return w/t-(w/t)*(w/t)

h=0.1
a,b = 1, 2
n = (b-a)/h
t = a
w = 1
Y = []
for i in range(int(n)):
    Y.append(w)
    w += h*f(t, w)
    t = a+i*h

fig = plt.figure(figsize=(12, 7))  
ax = fig.add_subplot(1, 1, 1)  
x = np.linspace(a,b, n)
ax.plot(x, Y,'*',label='euler')
plt.legend(loc=4)
# plt.show()
print(x,Y)