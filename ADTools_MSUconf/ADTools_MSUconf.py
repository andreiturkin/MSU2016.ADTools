"""
This Python file can be used to replicate the results obtained for:
A.Turkin "Automatic differentiation in Python"
Author: Andrei Turkin.
E-mail: andrei_turkin@hotmail.com
"""
####################################
#Automatic Differentiation Tools
####################################
from LJ_PyAdolc import PyAdolc_dvLJ, PyAdolc_Init
from LJ_PyCppAD import PyCppAD_dvLJ, PyCppAD_Init
from LJ_CasADi import CasADi_dvLJ, CasADi_Init
from LJ_CGT import CGT_dvLJ, CGT_Init
from LJ_Theano import Theano_dvLJ, Theano_Init
from LJ_AD import AD_dvLJ, AD_Init

####################################
#Additional Tools
####################################
from ExcelTools import SaveSpeedtoExcelFile
from DatasetTools import GetXFromDataSet

####################################
import timeit
###################################
from itertools import chain

#Number of dimensions
D = 3
#Functions
possibles = 0

def SpeedTest():
    global N
    global possibles
    
    ListOfGradFunc1 = ['PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ', 'CGT_dvLJ', 'Theano_dvLJ', 'AD_dvLJ' ]
    ListOfGradFunc2 = ['PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ', 'CGT_dvLJ']
    ListOfGradFunc3 = ['PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ']
    
    #Find all of the functions by their names
    possibles = globals().copy()
    possibles.update(locals())

    # Initialize the initial matrix of atom positions
    ListOfXs = GetXFromDataSet('Dataset/')
    
    myrange = chain(range(4,40,4),range(40,400,10),range(400,len(ListOfXs),50))
    
    for N in myrange:
        x = ListOfXs[N-3]
        print '\nInitial condition: N={}\n'.format(N)
        print x
        
        if N<28:
            FunctionList = ListOfGradFunc1
            TimeValues = RoundExec(ListOfGradFunc1, x)
        elif N>=28 and N<300:
            FunctionList = ListOfGradFunc2
            TimeValues = RoundExec(ListOfGradFunc2, x)
        else:
            FunctionList = ListOfGradFunc3
            TimeValues = RoundExec(ListOfGradFunc3, x)
        SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, TimeValues)

def RoundExec(FunctionList,x):
    ListOfTimes = []
    for i in range(len(FunctionList)):
        InitFunc = possibles.get('{}_Init'.format(FunctionList[i].split('_')[0]))
        InitFunc(x)
        MainFunc = possibles.get(FunctionList[i])
        t = timeit.Timer(lambda: MainFunc(x))
        exectime = t.timeit(number=3)/3
        print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
        ListOfTimes.append(exectime)
    return ListOfTimes
    
if __name__ == '__main__':
    SpeedTest()
    
    
