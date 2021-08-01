from scipy import integrate
def f(x):
    return x + 1
v, err = integrate.quad(f, 1, 2)
print(v)
print(err)