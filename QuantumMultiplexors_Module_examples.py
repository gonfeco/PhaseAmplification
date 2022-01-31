"""
Authors: Juan Santos Suárez & Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates implemented in the QuantumMultiplexors_Module
"""

import numpy as np
from AuxiliarFunctions import TestBins, PostProcessResults,RunJob



def LoadProbabilityProgram(p_X):
    """
    Creates a Quantum Program for loading an input numpy array with a 
    probability distribution with Quantum Multiplexors (QM).

    Parameters
    ----------

    p_X : numpy array
        Probability distribution of size m. Mandatory: m=2^n where n 
        is the number qbits of the quantum circuit. 

    Returns
    ----------
    
    Qprog: QLM Program.
        Quantum Program for loading input probability using QM
    P_gate: QLM AbstractGate
        Customized AbstractGate for loading input probability using QM
    """
    
    from QuantumMultiplexors_Module import LoadP_Gate
    P_gate = LoadP_Gate(p_X)
    from qat.lang.AQASM import Program
    Qprog = Program()
    qbits = Qprog.qalloc(P_gate.arity)
    Qprog.apply(P_gate, qbits)
    return Qprog, P_gate

def LoadIntegralProgram(f_X):
    """
    Creates a Quantum Program for loading the integral of an input 
    function given as a numpy array using Quantum Multiplexors (QM).

    Parameters
    ----------

    f_X : numpy array
        Function evaluation of size m. Mandatory: m=2^n where n is the
        number qbits of the quantum circuit. 

    Returns
    ----------
    
    Qprog: QLM Program
        Quantum Program for loading integral of the input function
    R_gate: QLM AbstractGate
        Customized AbstractGate for loading integral using QM
    """
    from QuantumMultiplexors_Module import LoadR_Gate
    R_gate = LoadR_Gate(f_X)
    from qat.lang.AQASM import Program, H
    Qprog = Program()
    qbits = Qprog.qalloc(R_gate.arity)
    for i in range(len(qbits)-1):
        Qprog.apply(H, qbits[i])
    Qprog.apply(R_gate, qbits)
    return Qprog, R_gate

def ExpectationLoadingData(p_X, f_X):
    """
    Creates a Quantum Program for loading mandatory data in order to
    load the expected value of a function f(x) over a x following a
    probability distribution p(x) using Quantum Multiplexors (QM).

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
    P_gate: QLM AbstractGate
        Customized AbstractGate for loading input probability using QM
    R_gate: QLM AbstractGate
        Customized AbstractGate for loading integral using QM
    """

    #Testing input
    assert len(p_X) == len(f_X), 'Arrays lenght are not equal!!'
    from QuantumMultiplexors_Module import LoadP_Gate, LoadR_Gate
    P_gate = LoadP_Gate(p_X)
    R_gate = LoadR_Gate(f_X)
    
    from qat.lang.AQASM import Program
    Qprog = Program()
    #The R gate have more qbits
    qbits = Qprog.qalloc(R_gate.arity)
    #Load Probability
    Qprog.apply(P_gate, qbits[:-1])
    #Load integral on the last qbit
    Qprog.apply(R_gate, qbits)
    return Qprog, P_gate, R_gate 


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
        qprog,_ = LoadProbabilityProgram(p_X)
    elif function == 'R':
        print('\t Load Integral')
        qprog,_ = LoadIntegralProgram(f_X)
    else:
        print('\t Load Complete Data')
        qprog,_,_ = ExpectationLoadingData(p_X, f_X)

    print('Making Circuit')
    circuit = qprog.to_circ()
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
    elif function == 'R':
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
    parser.add_argument('-t', '--type', default = None, help='Type of Loading: P: Load Probability. R: Load Integral. Otherwise: Load Complete Data')
    args = parser.parse_args()
    #print(args)

    Do(n_qbits=args.nqbits, depth=args.depth,function=args.type)







