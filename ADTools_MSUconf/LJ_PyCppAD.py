"""
Derivative Evaluation with PYADOLC for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""

####################################
#Automatic Differentiation Tools
####################################
#PyCppAD
from pycppad import *
import numpy as np
from scipy import optimize

D = 3
PyCppAD_Grad = 0
def PyCppAD_Init(x):
    global PyCppAD_Grad
    N = len(x)
    ix = np.zeros(np.ravel(x).shape)
    ad_x = independent(ix)
    # computing the function f: R^(NxD) -> R with PyCppAD
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((ad_x[i*D:i*D+D] - ad_x[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0)) 
    vLJt = np.array([4*vLJt])
    PyCppAD_Grad =  adfun(ad_x, vLJt)

def PyCppAD_dvLJ(x):
    return PyCppAD_Grad.jacobian(np.ravel(x))