from scipy.interpolate  import lagrange
import numpy as np
def f(x):
    return (x/9.0)*np.exp(x*x/(-1*18))
x = [1,4,7,10]
y = [0.1051,0.1827,0.0511,0.0043]
ret = lagrange(x,y)
print(ret.c)
print(ret)


#coding:utf-8
# import numpy as np
# from scipy.optimize import minimize_scalar
# def f(x):
#     return (x - 2) * x * (x + 2)**2
# res =_scalar(f, method='brent')
# print(res.x)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,6))
# x = np.linspace(-4,4,100)
# y = f(x)
# t = np.linspace(f(res.x),f(4),100)
# plt.plot([res.x] * len(x),t,color="red",label= "$x = res.x$",linewidth=2)
# plt.plot(x,y,color="orange",label="$x(x - 2)(x + 1)^2$",linewidth=2)
# plt.legend()
# plt.show()