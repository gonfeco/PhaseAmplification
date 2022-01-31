
"""Loading data routines

This module contains all the functions in order to load data into the
quantum state. Base of this module is the Lov Grover and Terry Rudolph
2008 papper:

    Creating superpositions that correspond to efficiently integrable
    probability distributions
    http://arXiv.org/abs/quant-ph/0208112v1

In this papper the probability is loaded by using a lot of controlled
Y Rotations. This kind of implementation is inneficient and a based 
quantum multiplexors one is prefered.

Author: Gonzalo Ferro Costas
Version: Initial version

"""

import numpy as np
from qat.lang.AQASM import QRoutine, AbstractGate, X, RY 
from AuxiliarFunctions import TestBins, LeftConditionalProbability
from AuxiliarFunctions import get_histogram


def CRBS_generator(Nqbits, ControlState, Theta):
    """ 
    This functions codify a input ControlState using Nqbits qbits and
    apply a controlled Y-Rotation by ControlState of Theta on one 
    aditional qbit.

    Parameters
    ----------

    Nqbits : int
        Number of qbits needed for codify the ControlState. 
    ControlState : int
        State for controlling the of the controlled Rotation.
    Theta : float
        Rotation angle (in radians)

    Returns
    ----------
    
    Qrout : quantum routine.
        Routine for creating a controlled multistate Rotation.

    """
    from qat.lang.AQASM import QRoutine, RY
    Qrout = QRoutine()
    #Use of quantum integer types for control qbits
    from qat.lang.AQASM.qint import QInt
    qcontrol = Qrout.new_wires(Nqbits, QInt)
    #Qbit where rotation should be applied
    qtarget = Qrout.new_wires(1)
    #The control qbits should be equal to the input ControlState
    #integer in order to apply the Rotation to the target qbit
    expresion = (qcontrol==ControlState)
    #An auxiliar qbit is created for storing the result of the 
    #expresion. This qbit will be in state |0> unless the control qbits
    #equals the Integer ControlState where state will change to |1>
    with Qrout.compute():
        qAux = expresion.evaluate()
    #The Rotation on the target qbit will be controlled by the auxiliar
    #qbit which contains the result of the expresion evaluation
    Qrout.apply(RY(Theta).ctrl(), qAux, qtarget)
    #Finally we need to undo the evaluation of the expresion in order to
    #get the original control qbits
    Qrout.uncompute()
    return Qrout

from qat.lang.AQASM import AbstractGate
#Using generator function an abstract gate is created
CRBS_gate = AbstractGate(
    "CRBS", 
    [int, int, float], 
    circuit_generator = CRBS_generator,
    arity = lambda x, y, z: x+1
)



def LoadP_Gate(ProbabilityArray):
    """
    Creates a customized AbstractGate for loading a discretized
    Probability.

    Parameters
    ----------

    ProbabilityArray : numpy array
        Numpy array with the discretized probability to load. The number
        of qbits will be log2(len(ProbabilityArray)). 


    Raises
    ----------
    AssertionError
        if len(ProbabilityArray) != 2^n 

    Returns
    ----------

    P_Gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array 
    """
    def P_generator():
        """
        Function generator for the AbstractGate that allows the loading 
        of a discretized Probability in a Quantum State.
    
        Returns
        ----------

        Qrout : Quantum Routine
            Quantum Routine for loading Probability
        """
    
        #ProbabilityArray = Dictionary['array']
        n_qbits = TestBins(ProbabilityArray, 'Probability')
    
        Qrout = QRoutine()
        qbits = Qrout.new_wires(n_qbits)
        nbins = len(ProbabilityArray)
    
        for i in range(0, n_qbits):
            ConditionalProbability = LeftConditionalProbability(
                i, ProbabilityArray)
            Thetas = 2.0*(np.arccos(np.sqrt(ConditionalProbability)))
    
            if i == 0:
                #The first qbit is a typical y Rotation
                Qrout.apply(RY(Thetas[0]), qbits[0])
            else:
                #The different rotations should be applied over the 
                #i+1 qbit. Each rotation is controlled by all the 
                #posible states formed with i qbits
                for j, theta in enumerate(Thetas):
                    #Next lines do the following operation: 
                    #|j> x Ry(2*\theta_{j})|0>
                    gate = CRBS_gate(i, j, theta)
                    Qrout.apply(gate, qbits[:i+1])
        return Qrout
    P_Gate = AbstractGate(
        "P_Gate",
        [],
        circuit_generator = P_generator,
        arity = TestBins(ProbabilityArray, 'Probability')
    )
    return P_Gate()

def LoadR_Gate(FunctionArray):
    """
    Creates a customized AbstractGate for loading the integral of a 
    discretized function in a Quantum State.
    
    Parameters
    ----------

    FunctionArray : numpy array 
        Numpy array with the discretized function to load. 
        The number of qbits will be log2(len(FunctionArray))+1.
        Integral will be load in the last qbit 

    Raises
    ----------
    AssertionError
        if len(FunctionArray) != 2^n 

    Returns
    ----------
    
    R_Gate: AbstractGate
        AbstractGate customized for loadin the integral of the function.
    """
    def R_generator():
        """
        Function generator for creating an AbstractGate that allows 
        the loading of the integral of a given discretized function 
        array into a Quantum State.
    
        Returns
        ----------

        Qrout : quantum routine
            Routine for loading the input function as a integral
            on the last qbit. 
        """
        #FunctionArray = Dictionary['array']
        nqbits_ = TestBins(FunctionArray, 'Function')
        #Calculation of the rotation angles
        Thetas = 2.0*np.arcsin(np.sqrt(FunctionArray))
    
        Qrout = QRoutine()
        qbits = Qrout.new_wires(nqbits_+1)
        NumberOfStates = 2**nqbits_
        #Loop over the States
        for i in range(NumberOfStates):
            #State |i>
            #Generation of a Controlled rotation of theta by state |i>
            controlledR_gate = CRBS_gate(nqbits_, i, Thetas[i])
            Qrout.apply(controlledR_gate, qbits)
        return Qrout

    R_Gate = AbstractGate(
        "R_Gate",
        [],
        circuit_generator = R_generator,
        arity = TestBins(FunctionArray, 'Function')+1
    )
    return R_Gate()

