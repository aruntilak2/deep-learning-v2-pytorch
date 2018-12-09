import numpy as np
import matplotlib
a = np.array([0.2, 0.37, 0.03, 0.2, 0.1])
# a = np.arange(15).reshape(5,3)
print(a)

shape= a.shape
print(shape)

dim = a.ndim
print(dim)

typename= a.dtype.name
print(typename)

b= np.arange(10,30,0.5)
print(b)
e= np.arange(1,10)
print(e)
expo= np.exp(e)
print(expo)
log =np.log10(e)
print(log)
