"""
Derivative Evaluation with CGT for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""
import cgt
import numpy as np
from scipy import optimize

D = 3
CGT_Grad = 0

def CGT_Init(x):
    global CGT_Grad
    N = len(x)
    xt = cgt.vector('xt')
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((xt[i*D:i*D+D] - xt[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0))
    
    dvLJc = cgt.grad(4*vLJt, xt)    
    CGT_Grad = cgt.function([xt],dvLJc)
    
def CGT_dvLJ(x):
    return CGT_Grad(np.ravel(x))