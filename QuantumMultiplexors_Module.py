"""
Authors: Juan Santos Suárez & Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This module contains all the functions in order to load data into the quantum state using 
quantum multiplexors.

This module is based in the papper: 
V.V. Shende, S.S. Bullock, and I.L. Markov. Synthesis of quantum-logic circuits.
IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 
25(6):1000–1010, Jun 2006

arXiv:quant-ph/0406176v5

"""

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

def multiplexor_RY_m_recurs(qprog, qbits, thetas, m, j, sig = 1.):
    """ 
    Auxiliary function to create the recursive part of a multiplexor that applies an RY gate
    
    qprog = Quantum Program in which we want to apply the gates
    qbits = Nmber of qubits of the quantum program
    thetas (np.ndarray) = numpy array containing the set of angles that we want to apply
    m   (int) = number of remaining controls
    j   (int) = index of the target qubits
    sig (float) = accounts for wether our multiplexor is being decomposed with its lateral CNOT at the right or at the left, even if that CNOT is not present because it cancelled out (its values can only be +1. and -1.)
    """
    assert isinstance(m, int), 'm must be an integer'
    assert isinstance(j, int), 'j must be an integer'
    assert sig == 1. or sig == -1., 'sig can only be -1. or 1.'
    if m > 1:
        # If there is more that one control, the multiplexor shall be decomposed.
        # It can be checked that the right way to decompose it taking into account the simplifications is as
        x_l = 0.5*np.array([thetas[i]+sig*thetas[i+len(thetas)//2] for i in range (len(thetas)//2)]) #left angles
        x_r = 0.5*np.array([thetas[i]-sig*thetas[i+len(thetas)//2] for i in range (len(thetas)//2)]) #right angles
        
        multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, 1.)
        qprog.apply(CNOT, qbits[j-m], qbits[j])
        multiplexor_RY_m_recurs(qprog, qbits, x_r, m-1, j, -1.)
        
        # Just for clarification, if we hadn't already simplify the CNOTs, the code should have been
        # if sign == -1.:
        #   multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, -1.)
        # qprog.apply(CNOT, qbits[j-m], qbits[j])
        # multiplexor_RY_m_recurs(qprog, qbits, x_r, m-1, j, -1.)
        # qprog.apply(CNOT, qbits[j-m], qbits[j])
        # if sign == 1.:
        #   multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, 1.)
        
    else: 
        # If there is only one control just apply the Ry gates
        ThetaPositive = (thetas[0]+sig*thetas[1])/2.0
        ThetaNegative = (thetas[0]-sig*thetas[1])/2.0
        qprog.apply(RY(ThetaPositive), qbits[j])
        qprog.apply(CNOT, qbits[j-1], qbits[j])
        qprog.apply(RY(ThetaNegative), qbits[j])
        
            
def multiplexor_RY_m(qprog, qbits, thetas, m, j):
    """
    Create a multiplexor that applies an RY gate on a qubit controlled by the former m qubits
    It will have its lateral cnot on the right.
    
    Given a 2^n vector of thetas this function creates a controlled by Y rotation of each theta. 
    The rotation is controlled by the basis state of a 2^n quantum system.
    If we had a n qbit system and a thetas = [thetas_0, thetas_1, ..., thetas_2^n-1] then the function applies
        - RY(thetas_0) controlled by state |0>_{n}
        - RY(thetas_1) controlled by state |1>_{n}
        - RY(thetas_2) controlled by state |2>_{n}
        - ...
        - RY(thetas_2^n-1) controlled by state |2^n-1>_{n}
    On the quantum system 

    Inputs:
        * qprog = Quantum Program in which we want to apply the gates
        * qbits = Nmber of qubits of the quantum program
        * thetas (np.ndarray) = numpy array containing the set of angles that we want to apply
        * m      (int) = number of remaining controls
        * j      (int) = index of the target qubits
    """
    multiplexor_RY_m_recurs(qprog, qbits, thetas, m, j)
    qprog.apply(CNOT, qbits[j-m], qbits[j])
    
def LoadProbability_Gate(ProbabilityArray, CentersArray):
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
    #assert Condition, 'Length of the ProbabilityArray must be of dimension 2^n with n a int. In this case is: {}.'.format(nqbits_)
    #
    #nqbits = int(nqbits_)
    #nbins = len(ProbabilityArray)
    nqbits = TestBins(ProbabilityArray, text='Function')

    
    P = AbstractGate("P", [int])
    def P_generator(nqbits):
        rout = QRoutine()
        reg = rout.new_wires(nqbits)
        print(reg)
        # Now go iteratively trough each qubit computing the probabilities and adding the corresponding multiplexor
        for m in range(nqbits):
            n_parts = 2**(m+1) #Compute the number of subzones which the current state is codifying
            edges = np.array([a+(b-a)*(i)/n_parts for i in range(n_parts+1)]) #Compute the edges of that subzones
        
            # Compute the probabilities of each subzone by suming the probabilities of the original histogram.
            # There is no need to compute integrals since the limiting accuracy is given by the original discretization.
            # Moreover, this approach allows to handle non analytical probability distributions, measured directly from experiments
            p_zones = np.array([np.sum(ProbabilityArray[np.logical_and(CentersArray>edges[i],CentersArray<edges[i+1])]) for i in range(n_parts)])
            # Compute the probability of standing on the left part of each zone 
            p_left = p_zones[[2*j for j in range(n_parts//2)]]
            # Compute the probability of standing on each zone (left zone + right zone)
            p_tot = p_left + p_zones[[2*j+1 for j in range(n_parts//2)]]
            
            # Compute the rotation angles
            thetas = 2.0*np.arccos(np.sqrt(p_left/p_tot))

            if m == 0:
                # In the first iteration it is only needed a RY gate
                rout.apply(RY(thetas[0]), reg[0])
            else:
                # In the following iterations we have to apply multiplexors controlled by m qubits
                # We call a function to construct the multiplexor, whose action is a block diagonal matrix of Ry gates with angles theta
                multiplexor_RY_m(rout, reg, thetas, m, m)
        return rout
    P.set_circuit_generator(P_generator)
    P_gate = P(nqbits)
    return P_gate

def LoadIntegralFunction_Gate(FunctionArray):
    """
    Load the values of the function f on the states in which the value of the auxiliary qubit is 1 once the probabilities are already loaded.
    The number of the qbits of the gate will be log2(len(FunctionArray)) + 1. This is mandatory. The integral will be loaded in last qbit
    
    Inputs:
        * FunctionArray: np.array. Discretized arrray with the function for integral loading
    Outputs:
        * R_gate (ParamGate) : gate that loads the function into the amplitudes
    """
    assert np.all(FunctionArray<=1.), 'The image of the function must be less than 1. Rescaling is required'
    assert np.all(FunctionArray>=0.), 'The image of the function must be greater than 0. Rescaling is required'
    assert isinstance(FunctionArray, np.ndarray), 'the output of the function p must be a numpy array'

    nqbits = TestBins(FunctionArray, text='Function')
    thetas = 2.0*np.arcsin(np.sqrt(FunctionArray))

    R = AbstractGate("R", [int])# + [float for theta in thetas])
    def R_generator(nqbits):#, *thetas):
        rout = QRoutine()
        reg = rout.new_wires(nqbits+1)
        multiplexor_RY_m(rout, reg, thetas, nqbits, nqbits)
        return rout
    R.set_circuit_generator(R_generator)
    R_gate = R(nqbits)
    return R_gate
