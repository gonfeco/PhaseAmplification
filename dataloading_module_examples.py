"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates in the dataloading_module
"""

import numpy as np
import pandas as pd


from AuxiliarFunctions import TestBins, PostProcessResults, RunJob


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
    #Creation of the AbstractGate LoadP_Gate
    from dataloading_module import LoadP_Gate
    #The probability should be given as a python dictionary with key array
    P_gate = LoadP_Gate(p_X)
    
    #Create the program
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits)
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
    #Creation of AbstractGate LoadR_Gate
    from dataloading_module import LoadR_Gate 
    #The function should be given as a python dictionary with key array
    R_gate = LoadR_Gate(f_X)

    #Create the program
    from qat.lang.AQASM import Program, H
    qprog = Program()
    #The additional qbit is where the integral will be encoded
    qbits = qprog.qalloc(nqbits+1)
    #Mixture state
    for i in range(nqbits):
        qprog.apply(H, qbits[i])
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
    from dataloading_module import LoadR_Gate, LoadP_Gate
    P_gate = LoadP_Gate(p_X)    
    R_gate = LoadR_Gate(f_X)
    
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
    print('########################################')
    print('#########Connection to QLMaSS###########')
    print('########################################')

    #QPU connection
    try:
        from qat.qlmaas import QLMaaSConnection
        connection = QLMaaSConnection('qlm')
        LinAlg = connection.get_qpu("qat.qpus:LinAlg")
        lineal_qpu = LinAlg()
    except (ImportError, OSError) as e:
        print('Problem: usin PyLinalg')
        from qat.qpus import PyLinalg
        lineal_qpu = PyLinalg()

    print('Creating Program')
    if function == 'P':
        print('\t Load Probability')
        qprog = LoadProbabilityProgram(p_X)
    elif function == 'I':
        print('\t Load Integral')
        qprog = LoadIntegralProgram(f_X)
    else:
        print('\t Load Complete Data')
        qprog = LoadingData(p_X, f_X)

    print('Making Circuit')
    circuit = qprog.to_circ()
    if function == 'P':
        job = circuit.to_job()
    else:
        job = circuit.to_job(qubits=[n_qbits])
    result = RunJob(lineal_qpu.submit(job))
    results = PostProcessResults(result)
    print(results)
    if function == 'P':
        Condition = np.isclose(results['Probability'], p_X).all()
        print('Probability load data: \n {}'.format(p_X))
        print('Probability Measurements: \n {}'.format(results['Probability']))
        print('This is correct? {}'.format(Condition))
    elif function == 'I':
        MeasurementIntegral = results['Probability'][1]*2**(n_qbits)
        print('Integral load data: {}'.format(sum(f_X)))
        print('Integral Measurement: {}'.format(MeasurementIntegral)) 
        Condition = np.isclose(MeasurementIntegral, sum(f_X))
        print('This is correct? {}'.format(Condition))
    else:
        MeasurementIntegral = results['Probability'][1]
        print('Integral Measurement: {}'.format(MeasurementIntegral)) 
        print('Expectation of f(x) for x~p(x): Integral p(x)f(x): {}'.format(sum(p_X*f_X)))
        Condition = np.isclose(MeasurementIntegral, sum(p_X*f_X))
        print('This is correct? {}'.format(Condition))


if __name__ == '__main__':
    "Working Example"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nqbits', type=int, help='Number Of Qbits', default  = 6)
    parser.add_argument('-depth', type=int, help='Depth of the Diagram', default = 0)
    parser.add_argument('-t', '--type', default = None, help='Type of Loading: P: Load Probability. I: Load Integral. Otherwise: Load Complete Data')
    args = parser.parse_args()
    #print(args)

    Do(n_qbits=args.nqbits, depth=args.depth,function=args.type)
