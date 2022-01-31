"""
This script contains several examples of use of the gates in the 
dataloading_module

Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

import numpy as np
import pandas as pd


from AuxiliarFunctions import TestBins, PostProcessResults, RunJob


def LoadProbabilityProgram(p_X):
    """
    Creates a Quantum Program for loading an input numpy array with a 
    probability distribution.

    Parameters
    ----------

    p_X : numpy array
        Probability distribution of size m. Mandatory: m=2^n where n 
        is the number qbits of the quantum circuit. 


    Returns
    ----------
    
    Qprog: QLM Program.
        Quantume Program for loading input probability
    """
    #Creation of the AbstractGate LoadP_Gate
    from dataloading_module import LoadP_Gate
    #The probability should be given as a python dictionary with key array
    P_gate = LoadP_Gate(p_X)
    
    #Create the program
    from qat.lang.AQASM import Program
    Qprog = Program()
    #Number of Qbits is defined by the arity of the Gate
    qbits = Qprog.qalloc(P_gate.arity)
    #Apply Abstract gate to the qbits
    Qprog.apply(P_gate, qbits)
    return Qprog


def LoadIntegralProgram(f_X):
    """
    Creates a Quantum Pogram for loading the integral of an input 
    function given as a numpy array

    Parameters
    ----------

    f_X : numpy array
        Function evaluation of size m. Mandatory: m=2^n where n is the
        number qbits of the quantum circuit. 

    Returns
    ----------
    
    Qprog: QLM Program
        Quantum Program for loading integral of the input function
    """
    #Qbits of the Quantum circuit depends on Function array length
    #nqbits = TestBins(f_X, 'Function')
    #Creation of AbstractGate LoadR_Gate
    from dataloading_module import LoadR_Gate 
    R_gate = LoadR_Gate(f_X)

    #Create the program
    from qat.lang.AQASM import Program, H
    Qprog = Program()
    #The number of qbits is defined by the arity of the gate
    nqbits = R_gate.arity
    qbits = Qprog.qalloc(nqbits)
    #Mixture of the controlled states (that are the first nqbits-1 qbits)
    for i in range(nqbits-1):
        Qprog.apply(H, qbits[i])
    #Apply Abstract gate to the qbits
    #Last qbit is where the integral of the function will be loaded
    Qprog.apply(R_gate, qbits)
    return Qprog

def ExpectationLoadingData(p_X, f_X):
    """
    Creates a Quantum Program for loading mandatory data in order to
    load the expected value of a function f(x) over a x following a
    probability distribution p(x).

    Parameters
    ----------

    p_X : numpy array
        Probability distribution of size m. Mandatory: m=2^n where n 
        is the number qbits of the quantum circuit. 
    f_X : numpy array
        Function evaluation of size m. Mandatory: m=2^n where n is the
        number qbits of the quantum circuit. 

    Returns
    ----------
    
    Qprog: QLM Program.
        Quantum Program for loading input probability
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
    Qprog = Program()
    qbits = Qprog.qalloc(nqbits+1)
    #Load Probability
    Qprog.apply(P_gate, qbits[:-1])
    #Load integral on the last qbit
    Qprog.apply(R_gate, qbits)
    return Qprog

def Do(n_qbits=6, depth=0, function='DataLoading'):
    """
    Function for testing purpouses. This function is used when the 
    script is executed from command line using arguments. It executes
    the three implemented fucntions of this script:
        * LoadProbabilityProgram
        * LoadIntegralProgram
        * ExpectationLoadingData

    Parameters
    ----------
    n_qbits : int.
        Number of Qbits for the quantum circuit. 
    depth : int
        Depth for visualizar the Quantum Circuit
    function : str
        String that indicates which of the before functions should be 
        used: 
            'P' : LoadProbabilityProgram 
            'I' : LoadIntegralProgram
            Otherwise : ExpectationLoadingData
    """
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
        Qprog = LoadProbabilityProgram(p_X)
    elif function == 'I':
        print('\t Load Integral')
        Qprog = LoadIntegralProgram(f_X)
    else:
        print('\t Load Data for Expected Value of function')
        Qprog = ExpectationLoadingData(p_X, f_X)

    print('Making Circuit')
    circuit = Qprog.to_circ()

    from qat.core.console import display
    display(circuit, max_depth = depth)
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
    parser.add_argument('-n', '--nqbits', type=int, help='Number Of Qbits', 
        default  = 6)
    parser.add_argument('-depth', type=int, help='Depth of the Diagram', 
        default = 0)
    parser.add_argument('-t', '--type', default = None, 
        help='Type of Loading: P: Load Probability. I: Load Integral.\
        Otherwise: Load Complete Data')
    args = parser.parse_args()
    #print(args)

    Do(n_qbits=args.nqbits, depth=args.depth,function=args.type)
