"""
Derivative Evaluation with CasADi for the following function

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
#CasADi
from casadi import *
import numpy as np
from scipy import optimize

D = 3
CasADi_Grad = 0
def CasADi_Init(x):
    global CasADi_Grad
    N = len(x)
    xc = SX.sym('xc',1,x.size)
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = 0.0
            for d in range(D):
                rho += (xc[i*D+d] - xc[j*D+d])**2
            vLJ += rho**(-6.0)-(rho**(-3.0))
    F = Function('F',[xc],[4*vLJ])
    CasADi_Grad = F.jacobian()

def CasADi_dvLJ(x):
    return CasADi_Grad.call([x.ravel()])[0]
