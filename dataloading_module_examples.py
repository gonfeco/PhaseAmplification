"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates in the dataloading_module
"""

import numpy as np
import pandas as pd


from AuxiliarFunctions import TestBins, PostProcessResults


def LoadProbabilityProgram(p_X):
    """
    Creates a Quantum Program for loading an input numpy array with a probability distribution.
    Inputs:
        * p_X: np.array. Probability distribution of size m. Mandatory: m=2^n where n is the number
        qbits of the quantum circuit. 
    Outputs:
        * qprog: qlm program for loading input probability
    """
    #Qbits of the Quantum circuit depends on Probability length
    nqbits = TestBins(p_X, 'Probability')
    
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits)
    #Creation of P_gate
    from dataloading_module import CreatePG
    P_gate = CreatePG(p_X)
    #Apply Abstract gate to the qbits
    qprog.apply(P_gate, qbits)
    return qprog


def LoadIntegralProgram(f_X):
    """
    Creates a Quantum Circuit for loading the integral of the input numpy array with a function evaluation 
    Inputs:
        * f_X: np.array. Function evaluation of size m. Mandatory: m=2^n where n is the number
        qbits of the quantum circuit. 
    Outputs:
        * program: qlm program for loading integral of the input function
    """
    #Qbits of the Quantum circuit depends on Function array length
    nqbits = TestBins(f_X, 'Function')
    
    from qat.lang.AQASM import Program, H
    qprog = Program()
    #The additional qbit is where the integral will be encoded
    qbits = qprog.qalloc(nqbits+1)
    for i in range(nqbits):
        qprog.apply(H, qbits[i])
    #Creation of P_gate
    from dataloading_module import CreateLoadFunctionGate
    R_gate = CreateLoadFunctionGate(f_X)
    #Apply Abstract gate to the qbits
    qprog.apply(R_gate, qbits)
    return qprog

def LoadingData(p_X, f_X):
    """
    Load all the mandatory data to load in a quantum program the expected value 
    of a function f(x) over a x following a probability distribution p(x).
    Inputs:
        * p_X: np.array. Array of the discretized probability density
        * f_X: np.array. Array of the discretized funcion
    Outpus:
        * qprog: quantum program for loading the expected value of f(x) for x following a p(x) distribution
    """
    #Testing input
    nqbits_p = TestBins(p_X, 'Probability')
    nqbits_f = TestBins(f_X, 'Function')
    assert nqbits_p == nqbits_f, 'Arrays lenght are not equal!!'
    nqbits = nqbits_p
    
    #Creation of Gates
    from dataloading_module import CreatePG
    P_gate = CreatePG(p_X)    
    from dataloading_module import CreateLoadFunctionGate
    R_gate = CreateLoadFunctionGate(f_X)
    
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits+1)
    #Load Probability
    qprog.apply(P_gate, qbits[:-1])
    #Load integral on the last qbit
    qprog.apply(R_gate, qbits)
    return qprog

def Do(n_qbits=6, depth=0, function='DataLoading'):
    def p(x):
        return x*x
    def f(x):
        return np.sin(x)
    #The number of bins 
    m_bins = 2**n_qbits
    LowerLimit = 0.0
    UpperLimit = 1.0 
    from AuxiliarFunctions import  get_histogram, PostProcessResults
    X, p_X = get_histogram(p, LowerLimit, UpperLimit, m_bins)
    f_X = f(X)
    print('Creating Program')
    qprog = LoadIntegralProgram(f_X)
    print('Making Circuit')
    circuit = qprog.to_circ()
    job = circuit.to_job()
    print(job)
    print('########################################')
    print('#########Connection to QLMaSS###########')
    print('########################################')
    from qat.qlmaas import QLMaaSConnection
    connection = QLMaaSConnection()
    LinAlg = connection.get_qpu("qat.qpus:LinAlg")
    lineal_qpu = LinAlg()
    result = lineal_qpu.submit(job)
    R_results = PostProcessResults(result.join())
    print(R_results)

if __name__ == '__main__':
    "Working Example"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nqbits', type=int, help='Number Of Qbits', default  = 6)
    parser.add_argument('-depth', type=int, help='Depth of the Diagram', default = 0)
    args = parser.parse_args()

    Do(n_qbits=args.nqbits, depth=args.depth)


