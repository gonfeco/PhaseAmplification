
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

def P_generator(Dictionary):
    """
    Function generator for the AbstractGate that allows the loading of a discretized Probability in a Quantum State.
    Inputs:
        * ProbabilityArray: dict. Python dictionary whit a key named "array" whose corresponding item is a numpy array with the discretized
    probability to load. If ProbabilityArray = Dictionary['array']. The number of qbits will be log2(len(ProbabilityArray)). 
    Outuput:
        * qrout: Quantum routine. Routine for loading the discrete probability.
    """

    ProbabilityArray = Dictionary['array']
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


LoadP_Gate = AbstractGate(
    "P_Gate",
    [dict],
    circuit_generator = P_generator,
    arity = lambda x:TestBins(x['array'], 'Function')
)

#def CreatePG(ProbabilityArray):
#    """
#    Given a discretized probability array the function creates a AbstracGate that allows the load
#    of the probability in a Quantum State. The number of qbits of the gate will be log2(len(ProbabilityArray))
#    Inputs:
#    * ProbabilityArray: np.array. Discretized arrray with the probability to load
#    Outuput:
#    * P_gate: Abstract Gate. Gate for loading Input probability in a quantum state
#    """
#    
#    #Number of Input qbits for the QWuantum Gate
#    #nqbits_ = np.log2(len(ProbabilityArray))
#    ##Probability array must have a dimension of 2^n.
#    #Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
#    #if Condition == False:
#    #    raise ValueError(
#    #        'Length of the ProbabilityArray must be of dimension 2^n with n a int. In this case is: {}.'.format(
#    #            nqbits_
#    #        )
#    #    )
#    #nqbits_ = int(nqbits_)
#    nqbits_ = TestBins(ProbabilityArray, 'Probability')
#    def LoadProbability_generator(NumbeOfQbits):
#        
#        qrout = QRoutine()
#        qbits = qrout.new_wires(NumbeOfQbits)
#        nbins = 2**NumbeOfQbits        
#        
#        #Iteratively generation of the circuit
#        for i in range(0, NumbeOfQbits):
#            #Each step divides the bins in the step before by 2:
#            #if i=1 -> there are 2 divisions so the step splits each one in 2 so 4 new bins are generated
#            #if i=2 -> there are 4 divisions so the step split each one in 2 so 8 new bins are generated
#            
#            #Calculates Conditional Probability
#            ConditionalProbability = LeftConditionalProbability(i, ProbabilityArray)
#            #Rotation angles: length: 2^(i-1)-1 and i the number of qbits of the step
#            Thetas = 2.0*(np.arccos(np.sqrt(ConditionalProbability)))
#
#            if i == 0:
#                #The first qbit is a typical y Rotation
#                qrout.apply(RY(Thetas[0]), qbits[0])
#            else:
#                #The different rotations should be applied  over the i+1 qbit.
#                #Each rotation is controlled by all the posible states formed with i qbits
#                for j, theta in enumerate(Thetas):
#                    #Next lines do the following operation: |j> x Ry(2*\theta_{j})|0>
#                    gate = CRBS_gate(i, j, theta)
#                    qrout.apply(gate, qbits[:i+1])    
#        return qrout
#    
#    LoadP_Gate = AbstractGate("P_Gate", [int])   
#    LoadP_Gate.set_circuit_generator(LoadProbability_generator)
#    #Now We generated the complete Quantum Gate
#    P_gate = LoadP_Gate(nqbits_)
#    return P_gate   

def R_generator(Dictionary):
    """
    Function generator for creating an AbstractGate that allows the loading of the integral of a given discretized function array
    into a Quantum State.
    Inputs:
        * Dictionary: dict. Python dictionary with a key named "array" whose corresponding item is a numpy array with the discrietized function. If the discretized function is FunctionArray = Dictionary['array'] the number of qbits will be log2(len(FunctionArray)) + 1 qbits.
    Outuput:
        * qrout: quantum routine. Routine for loading the input function as a integral on the last qbit. 
    """
    FunctionArray = Dictionary['array']
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

LoadR_Gate = AbstractGate(
    "R_Gate",
    [dict],
    circuit_generator = R_generator,
    arity = lambda x:TestBins(x['array'], 'Function')+1
)


#def CreateLoadFunctionGate(FunctionArray):
#    """
#    Given a discretized function array this function creates a AbstracGate that allows the load
#    of the functions in a Quantum State. The gate will have  log2(len(FunctionArray)) + 1 qbits. 
#    The first log2(len(FunctionArray)) qbits contain the state of the system.
#    Each posible measurement-state of the system applies a controlled rotation on the last qbit.
#    Inputs:
#    * FunctionArray: np.array. Discretized arrray with the function to load
#    Outuput:
#    * R_gate: AbstractGate. Gate for loading the input function as a integral on the last qbit. 
#    """
#    
#    #Number of qbits to codify Input Function
#    #nqbits_ = np.log2(len(FunctionArray))
#    ##FunctionArray array must have a dimension of 2^n.
#    #Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
#    #if Condition == False:
#    #    raise ValueError(
#    #        'Length of the ProbabilityArray must be of dimension 2^n with n a int. In this case is: {}.'.format(nqbits)
#    #    )
#    #nqbits_ = int(nqbits_)
#    nqbits_ = TestBins(FunctionArray, 'Function')
#
#    #Calculation of the rotation angles
#    Thetas = 2.0*np.arcsin(np.sqrt(FunctionArray))
#    
#    def LoadFunction_generator(NumbeOfQbits):
#        qrout = QRoutine()
#        #The function will be load in the additional qbit
#        qbits = qrout.new_wires(NumbeOfQbits+1)
#        NumberOfStates = 2**NumbeOfQbits
#        #Loop over the States
#        for i in range(NumberOfStates):
#            #State |i>
#            
#            #Generation of a Controlled rotation of theta by state |i>
#            controlledR_gate = CRBS_gate(NumbeOfQbits, i, Thetas[i])    
#            qrout.apply(controlledR_gate, qbits)
#        return qrout
#    
#    LoadF_Gate = AbstractGate("R_Gate", [int])
#    LoadF_Gate.set_circuit_generator(LoadFunction_generator)
#    R_gate = LoadF_Gate(nqbits_)
#    return R_gate    



def ExampleFunction(nqbits = 8, depth = 0):
    nbins = 2**nqbits
    a = 0
    b = 1
    #Probability function
    def p(x):
        return x*x
    #Discretized Probability to load
    centers, probs = get_histogram(p, a, b, nbins)
    #Create Quantum Circuit for probability loading
    from qat.lang.AQASM import Program
    qprog = Program()
    qbits = qprog.qalloc(nqbits)
    P_gate = CreatePG(probs)
    qprog.apply(P_gate, qbits)

    #Create the circuit from the program
    circuit = qprog.to_circ()
    
    #Display the circuit
    from qat.core.console import display
    display(circuit, max_depth = depth)

    #Create a Job from the circuit
    job = circuit.to_job()
    
    #Import and create the linear algebra simulator
    from qat.qpus import LinAlg
    linalgqpu = LinAlg()
    
    #Submit the job to the simulator LinAlg and get the results
    result = linalgqpu.submit(job)
    
    QP = []
    #Print the results
    for i, sample in enumerate(result):
        print("State %s probability %s corresponding probability %f" % (sample.state, sample.probability, probs[i]))
        QP.append(sample.probability)
    QP =np.array(QP)            
    #print(''.format(np.isclose(QP, probs)))
    print('Todo OK?: {}'.format(np.isclose(QP, probs).all()))


if __name__ == '__main__':
    "Working Example"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nqbits', type=int, help='Number Of Qbits', default  = 2)
    parser.add_argument('-depth', type=int, help='Depth of the Diagram', default = 0)
    args = parser.parse_args()
    #print(args.nqbits)
    ExampleFunction(args.nqbits, args.depth)
#Configuración del algoritmo
