import numpy as np

a= np.arange(1,5)
print(a)

def softmax(L):
    expL = np.exp(L)
    print(expL)
    sumExpL = sum(expL)
    print(sumExpL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

softmax(a)
print(softmax(a))
    