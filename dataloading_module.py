
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


def LeftConditionalProbability(InitialBins, Probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry Rudolph 2008 papper:
    'Creating superpositions that correspond to efficiently integrable probability distributions'
    http://arXiv.org/abs/quant-ph/0208112v1
    Given a discretized probability and an initial spliting the function splits each initial region in
    2 equally regions and calculates the condicional probabilities for x is located in the left part
    of the new regions when x is located in the region that contains the corresponding left region
    Inputs:
    * InitialBins: int. Number of initial bins for spliting the input probabilities
    * Probability: np.array. Array with the probabilities to be load. 
    InitialBins <= len(Probabilite)
    Outputs:
    * Prob4Left: conditional probabilities of the new InitialBins+1 splits    
    """
    #Initial domain division
    DomainDivisions = 2**(InitialBins)
    
    if DomainDivisions >= len(Probability):
        raise ValueError('The number of Initial Regions (2**InitialBins) must be lower than len(Probability)')
    
    #Original number of bins of the probability distribution
    nbins = len(Probability)
    #Number of Original bins in each one of the bins of Initial domain division 
    BinsByDomainDivision = nbins//DomainDivisions
    #Probability for x located in each one of the bins of Initial domain division
    Prob4DomainDivision = [
        sum(Probability[j*BinsByDomainDivision:j*BinsByDomainDivision+BinsByDomainDivision]) \
        for j in range(DomainDivisions)
    ]
    #Each bin of Initial domain division is splitted in 2 equal parts
    Bins4LeftDomainDivision = nbins//(2**(InitialBins+1))    
    
    #Probability for x located in the left bin of the new splits
    LeftProbs = [
        sum(Probability[j*BinsByDomainDivision:j*BinsByDomainDivision+Bins4LeftDomainDivision])\
        for j in range(DomainDivisions)
    ]    
    #Conditional probability of x located in the left bin when x is located in the 
    #bin of the initial domain division that contains the split
    #Basically this is the f(j) function of the article with j=0,1,2,...2^(i-1)-1 
    #and i the number of qbits of the initial domain division 
    Prob4Left = np.array(LeftProbs)/np.array(Prob4DomainDivision)    
    return Prob4Left

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
    cR = 'RY({})'.format(Theta) + '.ctrl()'*len(qcontrol)
    #The rotation is only applyied when qcontrol is in ControlState
    qrout.apply(eval(cR), qcontrol, qtarget)
    
    #Undo the operations for using the ControlState
    #for controlling the rotation
    for m, k in enumerate(bList):
        if k == False:
            qrout.apply(X,qcontrol[m])           
    return qrout    

#Using generator function an abstract gate is created
CRBS_gate = AbstractGate("CRBS_Gate", [int, int, float])   
CRBS_gate.set_circuit_generator(CRBS_generator)

def CreatePG(ProbabilityArray):
    """
    Given a discretized probability array the function creates a AbstracGate that allows the load
    of the probability in a Quantum State. The number of qbits of the gate will be log2(len(ProbabilityArray))
    Inputs:
    * ProbabilityArray: np.array. Discretized arrray with the probability to load
    Outuput:
    * P_gate: Abstract Gate. Gate for loading Input probability in a quantum state
    """
    
    #Number of Input qbits for the QWuantum Gate
    #nqbits_ = np.log2(len(ProbabilityArray))
    ##Probability array must have a dimension of 2^n.
    #Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
    #if Condition == False:
    #    raise ValueError(
    #        'Length of the ProbabilityArray must be of dimension 2^n with n a int. In this case is: {}.'.format(
    #            nqbits_
    #        )
    #    )
    #nqbits_ = int(nqbits_)
    nqbits_ = TestBins(ProbabilityArray, 'Probability')
    def LoadProbability_generator(NumbeOfQbits):
        
        qrout = QRoutine()
        qbits = qrout.new_wires(NumbeOfQbits)
        nbins = 2**NumbeOfQbits        
        
        #Iteratively generation of the circuit
        for i in range(0, NumbeOfQbits):
            #Each step divides the bins in the step before by 2:
            #if i=1 -> there are 2 divisions so the step splits each one in 2 so 4 new bins are generated
            #if i=2 -> there are 4 divisions so the step split each one in 2 so 8 new bins are generated
            
            #Calculates Conditional Probability
            ConditionalProbability = LeftConditionalProbability(i, ProbabilityArray)
            #Rotation angles: length: 2^(i-1)-1 and i the number of qbits of the step
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
    
    LoadP_Gate = AbstractGate("P_Gate", [int])   
    LoadP_Gate.set_circuit_generator(LoadProbability_generator)
    #Now We generated the complete Quantum Gate
    P_gate = LoadP_Gate(nqbits_)
    return P_gate   



def CreateLoadFunctionGate(FunctionArray):
    """
    Given a discretized function array this function creates a AbstracGate that allows the load
    of the functions in a Quantum State. The gate will have  log2(len(FunctionArray)) + 1 qbits. 
    The first log2(len(FunctionArray)) qbits contain the state of the system.
    Each posible measurement-state of the system applies a controlled rotation on the last qbit.
    Inputs:
    * FunctionArray: np.array. Discretized arrray with the function to load
    Outuput:
    * R_gate: AbstractGate. Gate for loading the input function as a integral on the last qbit. 
    """
    
    #Number of qbits to codify Input Function
    #nqbits_ = np.log2(len(FunctionArray))
    ##FunctionArray array must have a dimension of 2^n.
    #Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
    #if Condition == False:
    #    raise ValueError(
    #        'Length of the ProbabilityArray must be of dimension 2^n with n a int. In this case is: {}.'.format(nqbits)
    #    )
    #nqbits_ = int(nqbits_)
    nqbits_ = TestBins(FunctionArray, 'Function')

    #Calculation of the rotation angles
    Thetas = 2.0*np.arcsin(np.sqrt(FunctionArray))
    
    def LoadFunction_generator(NumbeOfQbits):
        qrout = QRoutine()
        #The function will be load in the additional qbit
        qbits = qrout.new_wires(NumbeOfQbits+1)
        NumberOfStates = 2**NumbeOfQbits
        #Loop over the States
        for i in range(NumberOfStates):
            #State |i>
            
            #Generation of a Controlled rotation of theta by state |i>
            controlledR_gate = CRBS_gate(NumbeOfQbits, i, Thetas[i])    
            qrout.apply(controlledR_gate, qbits)
        return qrout
    
    LoadF_Gate = AbstractGate("R_Gate", [int])
    LoadF_Gate.set_circuit_generator(LoadFunction_generator)
    R_gate = LoadF_Gate(nqbits_)
    return R_gate    


def get_histogram(p, a, b, nbin):
    """
    Given a function p, convert it into a histogram. The function must be positive, the normalization is automatic.
    Note that instead of having an analytical expression, p could just create an arbitrary vector of the right dimensions and positive amplitudes
    so that this procedure could be used to initialize any quantum state with real amplitudes
    
    a    (float)    = lower limit of the interval
    b    (float)    = upper limit of the interval
    p    (function) = function that we want to convert to a probability mass function. It does not have to be normalized but must be positive in the interval
    nbin (int)      = number of bins in the interval
    """
    step = (b-a)/nbin
    centers = np.array([a+step*(i+1/2) for i in range(nbin)]) #Calcula directamente los centros de los bines
    
    prob_n = p(centers)
    assert np.all(prob_n>=0.), 'Probabilities must be positive, so p must be a positive function'
    probs = prob_n/np.sum(prob_n)
    assert np.isclose(np.sum(probs), 1.), 'Probability is not getting normalized properly'
    return centers, probs

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
