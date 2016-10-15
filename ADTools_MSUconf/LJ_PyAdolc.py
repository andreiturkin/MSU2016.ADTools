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
#PyADOL-C
import adolc
import numpy as np
from scipy import optimize

D = 3

###################################
# PART 2b: Computation with PYADOLC
###################################
def Adolc_vLJ(x):
    N = len(x)/D
    vLJ = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((x[i*D:i*D+D] - x[j*D:j*D+D])**2).sum()
            vLJ += rho**(-6.0)-(rho**(-3.0))
    return 4*vLJ

def PyAdolc_Init(x):
    adolc.trace_on(0)
    ad_x = adolc.adouble(np.zeros(np.shape(np.ravel(x)),dtype=float))
    adolc.independent(ad_x)
    ad_y = Adolc_vLJ(ad_x)
    adolc.dependent(ad_y)
    adolc.trace_off()
    
def PyAdolc_dvLJ(x):
    return adolc.gradient(0, np.ravel(x))