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
#Theano
import theano
import theano.tensor as T

import numpy as np
from scipy import optimize

D = 3
Theano_Grad = 0
def Theano_Init(x):
    global Theano_Grad
    N = len(x)
    xt = T.dvector('xt')
    vLJt = theano.shared(0.0, name='vLJt')
    for j in range(1,N):
        for i in range(j):
            rho = ((xt[i*D:i*D+D] - xt[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0))
    
    dvLJt = T.grad(4*vLJt, xt)
    Theano_Grad = theano.function([xt],dvLJt)

def Theano_dvLJ(x):
    return Theano_Grad(np.ravel(x))