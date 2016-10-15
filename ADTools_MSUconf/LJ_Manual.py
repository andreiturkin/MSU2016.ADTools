"""
Derivative Evaluation for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""

import numpy as np
from scipy import optimize
D = 3
def Manual_Init(x):
    return 0
def Manual_dvLJ(x):
    N = len(x)
    g = np.zeros(np.shape(x),dtype=float)
    for n in range(N):
        for d in range(D):
            for m in range(N):
                if n != m:
                    g[n,d] -= 12*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**7 - 6*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**4
    return np.ravel(4*g)