"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates in the dataloading_module
"""

import numpy as np
import pandas as pd


from dataloading_module import  get_histogram

def TestBins(array, text='Probability'):
    """
    Testing Condition for numpy arrays. The length of the array must be 2^n with n an int.
    Inputs:
    """

    nqbits_ = np.log2(len(array))
    Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
    ConditionStr = 'Length of the {} Array must be of dimension 2^n with n an int. In this case is: {}.'.format(text, nqbits_)    
    assert Condition, ConditionStr
    return int(nqbits_)

def LoadProbabilityProgram(p_X):
    """
    Creates a Quantum Circuit for loading an input numpy array with a probability distribution.
    Inputs:
        * p_X: np.array. Probability distribution of size m. Mandatory: m=2^n where n is the number
        qbits of the quantum circuit. 
    Outputs:
        * circuit: qlm circuit for loading input probability
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
    #Creation of the Quantum Circuit
    circuit = qprog.to_circ()
    return circuit


def LoadIntegralProgram(f_X):
    """
    Creates a Quantum Circuit for loading the integral of the input numpy array with a function evaluation 
    Inputs:
        * f_X: np.array. Function evaluation of size m. Mandatory: m=2^n where n is the number
        qbits of the quantum circuit. 
    Outputs:
        * circuit: qlm circuit for loading integral of the input function
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
    #Creation of the Quantum Circuit
    circuit = qprog.to_circ()
    return circuit
