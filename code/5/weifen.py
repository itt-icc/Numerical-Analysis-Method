import numpy as np
from scipy.interpolate import lagrange
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
from scipy import integrate


def f1(x):
    return np.sin(x)*np.sin(x)-2*x*np.sin(x)+1

weifen = integrate.quad(f1,0.75,1.3)

print(weifen)

