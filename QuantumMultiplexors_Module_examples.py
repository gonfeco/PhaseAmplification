"""
Authors: Juan Santos Suárez & Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates implemented in the QuantumMultiplexors_Module
"""

import numpy as np
from AuxiliarFunctions import TestBins, PostProcessResults

def LoadProbabilityProgram(p_X):
    """
    Creates a Quantum Program for loading an input numpy array with a probability distribution.
    Inputs:
        * p_X: np.array. Probability distribution of size m. Mandatory: m=2^n where n is the number qbits of the quantum circuit. 
    Outputs:
        * qprog: qlm program for loading input probability
    """
    #Qbits of the Quantum circuit depends on Probability length
    nqbits = TestBins(p_X, 'Probability')
    
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits)
    #Creation of P_gate
    from QuantumMultiplexors_Module import LoadProbability_Gate
    P_gate = LoadProbability_Gate(p_X)
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
    from QuantumMultiplexors_Module import LoadIntegralFunction_Gate
    R_gate = LoadIntegralFunction_Gate(f_X)
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
    from QuantumMultiplexors_Module import LoadProbability_Gate
    P_gate = LoadProbability_Gate(p_X)
    from QuantumMultiplexors_Module import LoadIntegralFunction_Gate
    R_gate = LoadIntegralFunction_Gate(f_X)
    
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits+1)
    #Load Probability
    qprog.apply(P_gate, qbits[:-1])
    #Load integral on the last qbit
    qprog.apply(R_gate, qbits)
    return qprog