
"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This module contains all the functions in order to load data into the quantum state
Base of this module is the Lov Grover and Terry Rudolph 2008 papper:
'Creating superpositions that correspond to efficiently integrable probability distributions'
http://arXiv.org/abs/quant-ph/0208112v1
In this papper the probability is loaded by using a lot of controlled Y Rotations. 
This kind of implementation is inneficient and a based quantum multiplexors one is prefered.
"""

import numpy as np
from qat.lang.AQASM import QRoutine, AbstractGate, X, RY 
from AuxiliarFunctions import TestBins, LeftConditionalProbability, get_histogram


#CRBS = ControlledRotationByState
def CRBS_generator(N, ControlState, Theta):
    """
    This functions codify a input ControlState using N qbits and
    apply a controlled Rotation of an input angle Theta by the ControlState
    on one aditional qbit.
    Inputs:
    * N: int. Number of qbits needed for codify the ControlState. 
    * ControlState: int. State for controlling the of the controlled Rotation.
    * Theta: float. Rotation angle    
    """
    qrout = QRoutine()
    
    #Creates de control using first N
    qcontrol = qrout.new_wires(N)
    #An additional target qbit  
    qtarget = qrout.new_wires(1)    
    
    #Transform staje in binnary string
    bNumber = list(format(ControlState, '0{}b'.format(int(N))))
    #Binnary string to list of Booleans
    bList = [bool(int(i)) for i in bNumber]
    
    #This block contains the mandatory transformation to use the ControlState 
    #for performing a controlled Operation on the target qbit
    for m, k in enumerate(bList):
        if k == False:
            qrout.apply(X, qcontrol[m])
            
    #Apply the controlled rotation on the target qbit
    #The rotation is only applyied when qcontrol is in ControlState
    c_i_RY = RY(Theta).ctrl(len(qcontrol))
    qrout.apply(c_i_RY, qcontrol, qtarget)
    
    #Undo the operations for using the ControlState
    #for controlling the rotation
    for m, k in enumerate(bList):
        if k == False:
            qrout.apply(X,qcontrol[m])           
    return qrout    

#Using generator function an abstract gate is created
CRBS_gate = AbstractGate(
    "CRBS_Gate", 
    [int, int, float],
    circuit_generator = CRBS_generator,
    arity = lambda x,y,z: x+1

)   

def LoadP_Gate(ProbabilityArray):
    """
    Creates a customized AbstractGate for loading a discretized Probability
    Inputs:
        * ProbabilityArray: numpy array. Numpy array with the discretized probability to load. The number of qbits will be log2(len(ProbabilityArray)). 
    Output:
        * AbstractGate: AbstractGate customized 
    """
    def P_generator():
        """
        Function generator for the AbstractGate that allows the loading of a discretized Probability in a Quantum State.
        Output:
            * qrout: Quantum Routine
        """
    
        #ProbabilityArray = Dictionary['array']
        n_qbits = TestBins(ProbabilityArray, 'Probability')
    
        qrout = QRoutine()
        qbits = qrout.new_wires(n_qbits)
        nbins = len(ProbabilityArray)
    
        for i in range(0, n_qbits):
            ConditionalProbability = LeftConditionalProbability(i, ProbabilityArray)
            Thetas = 2.0*(np.arccos(np.sqrt(ConditionalProbability)))
    
            if i == 0:
                #The first qbit is a typical y Rotation
                qrout.apply(RY(Thetas[0]), qbits[0])
            else:
                #The different rotations should be applied  over the i+1 qbit.
                #Each rotation is controlled by all the posible states formed with i qbits
                for j, theta in enumerate(Thetas):
                    #Next lines do the following operation: |j> x Ry(2*\theta_{j})|0>
                    gate = CRBS_gate(i, j, theta)
                    qrout.apply(gate, qbits[:i+1])
        return qrout
    P_Gate = AbstractGate(
        "P_Gate",
        [],
        circuit_generator = P_generator,
        arity = TestBins(ProbabilityArray, 'Probability')
    )
    return P_Gate()

def LoadR_Gate(FunctionArray):
    """
    Creates a customized AbstractGate for loading the integral of discretized function in a Qunatum State.
    Inputs:
        * FunctionArray: numpy array. Numpy array with the discretized function to load. The number of qbits will be log2(len(FunctionArray))+1. Integral will be load in the last qbit 
    Output:
        * AbstractGate: AbstractGate customized 
    """
    def R_generator():
        """
        Function generator for creating an AbstractGate that allows the loading of the integral of a given discretized function array into a Quantum State.
        Outuput:
            * qrout: quantum routine. Routine for loading the input function as a integral on the last qbit. 
        """
        #FunctionArray = Dictionary['array']
        nqbits_ = TestBins(FunctionArray, 'Function')
        #Calculation of the rotation angles
        Thetas = 2.0*np.arcsin(np.sqrt(FunctionArray))
    
        qrout = QRoutine()
        qbits = qrout.new_wires(nqbits_+1)
        NumberOfStates = 2**nqbits_
        #Loop over the States
        for i in range(NumberOfStates):
            #State |i>
            #Generation of a Controlled rotation of theta by state |i>
            controlledR_gate = CRBS_gate(nqbits_, i, Thetas[i])
            qrout.apply(controlledR_gate, qbits)
        return qrout

    R_Gate = AbstractGate(
        "R_Gate",
        [],
        circuit_generator = R_generator,
        arity = TestBins(FunctionArray, 'Function')+1
    )
    return R_Gate()

