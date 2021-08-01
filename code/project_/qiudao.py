import numpy as np
from scipy.misc import derivative
def f(x):
    return x**5
for x in range(1, 4):
# 直接求导
    print(derivative(f, x, dx=1e-10))