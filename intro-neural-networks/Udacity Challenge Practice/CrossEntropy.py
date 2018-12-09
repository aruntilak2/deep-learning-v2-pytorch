import numpy as np
def cross_entropy(Y, P):
    Y = np.float_(Y)
    print(Y)
    P = np.float_(P)
    print(P)
    logY = np.log(Y)
    print(logY)
    logP = np.log(P)
    print(logP)
    z = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    print(z)
cross_entropy(0.6, 0.4)
