"""
Derivative Evaluation with Theano for the following function

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
#AD
from ad import gh

import numpy as np
from scipy import optimize

D = 3
AD_Grad = 0

def AD_vLJ_vec(x):
    N = len(x)/D
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = ((x[i*D:i*D+D] - x[j*D:j*D+D])**2).sum()
            vLJ += rho**(-6.0)-(rho**(-3.0))
    return 4*vLJ

def AD_Init(x):
    global AD_Grad
    AD_Grad = gh(AD_vLJ_vec)[0]
    
def AD_dvLJ(x):
    return AD_Grad(np.ravel(x))