"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

from qat.lang.AQASM import AbstractGate, QRoutine, X, Z
from qat.lang.AQASM import QRoutine, H, build_gate, Program

@build_gate("U_Phi_0", [int], arity=lambda n: n)
def UPhi0_Gate(nqbits):
    #def U_Phi0_generator(nqbits):
    """
    Circuit generator for creating gate U_Phi_0 for Phase 
    Amplification Algorithm. For a n+1 qbit system where the quantum 
    state can be decomposed by:
        |Psi>_{n+1} =a*|Phi_{n}>|1>+b|Phi_{n}>|0>. 
    The function implements a reflexion around the state |Phi_{n}>|1>. 
    The operator implemented is: 
        I-2|Phi_{n}>|0><0|<Phi_{n}|
    
    Parameters
    ----------

    nqbits : int
        Number of Qbits of the Abstract Gate

    Returns
    ----------

    Qrout : QLM Routine
        Quantum routine wiht the circuit implementation for operator:
            I-2|Phi_{n}>|0><0|<Phi_{n}|
    """
    Qrout = QRoutine()
    qbits = Qrout.new_wires(nqbits)
    Qrout.apply(X, qbits[-1])
    Qrout.apply(Z, qbits[-1])
    Qrout.apply(X, qbits[-1])
    
    return Qrout
##Definition of the Abstract Gate
#U_Phi_0 = AbstractGate(
#    "U_Phi_0",
#    [int],
#    circuit_generator = U_Phi0_generator,
#    arity = lambda x: x
#)

@build_gate("D_0", [int], arity=lambda n: n)
def D0_Gate(nqbits):
    """
    Circuit generator for create an Abstract Gate that implements a
    Reflexion around the state perpendicular to |0>_{n}.
    Implements operator: 
        I-2|0>_{n}{n}<0| = X^{n}c^{n-1}ZX^{n}
    
    Parameters
    ----------

    nqbits : int
        Number of Qbits of the Abstract Gate

    Returns
    ----------

    Qrout : QLM Routine
        Quantum routine wiht the circuit implementation for operator:
            I-2|0>_{n}{n}<0|
    """
    Qrout = QRoutine()
    qbits = Qrout.new_wires(nqbits)
    for i in range(nqbits):
        Qrout.apply(X, qbits[i])
    #Controlled Z gate by n-1 first qbits
    c_n_Z = Z.ctrl(nqbits-1)
    Qrout.apply(c_n_Z, qbits[:-1], qbits[-1])
    for i in range(nqbits):
        Qrout.apply(X, qbits[i])
    
    return Qrout
#D0_Gate = AbstractGate(
#    "D_0",
#    [int],
#    circuit_generator = D0_generator,
#    arity = lambda x: x
#)

def Load_UPhi_Gate(P_gate, R_gate):
    """
    Create gate U_Phi mandatory for Phase Amplification Algorithm.
    The operator to implement is: 
        I-2|Phi_{n-1}><Phi_{n-1}|. 
    Where the state |Phi_{n-1}> is: |Phi_{n-1}>=R*P*|0_{n+1}>. 
    Where R and P are the gates to load the integral of a function 
    f(x) and  the load of a distribution probabilitiy p(x) respectively.

    Parameters
    ----------

    P_gate : QLM AbstractGate
        Customized AbstractGate for loading probability distribution.
    R_gate : QLM AbstractGate
        Customized AbstractGatel for loading integral of a function f(x)
    Outputs:
        * U_Phi: guantum gate that implements U_Phi gate
    """
    
    from qat.lang.AQASM import AbstractGate, QRoutine
    #The arity of the R_gate fix the number of qbits for the routine
    nqbits = R_gate.arity

    @build_gate("UPhi", [], arity = nqbits)
    def UPhi_Gate():
        """
        Circuit generator for the UPhi_Gate.
        Operation to be implemented: R*P*D_0*P^{+}R^{+}
    
        Returns
        ----------
    
        Qrout : Quantum Routinequantum
            Quantum Routine with the circuit implementation for operator:
            R*P*D_0*P^{+}R^{+}
        """
        
        Qrout = QRoutine()
        qbits = Qrout.new_wires(nqbits)
        Qrout.apply(R_gate.dag(), qbits)
        Qrout.apply(P_gate.dag(), qbits[:-1])
        D_0 = D0_Gate(nqbits)
        Qrout.apply(D_0, qbits)
        Qrout.apply(P_gate, qbits[:-1])
        Qrout.apply(R_gate, qbits)
        return Qrout
    #U_Phi = AbstractGate(
    #    "UPhi", 
    #    [],
    #    circuit_generator = U_Phi_generator,
    #    arity = nqbits
    #)
    return UPhi_Gate()    

def Load_Q_Gate(P_gate, R_gate):
    """
    Create complete AbstractGate for Amplitude Amplification Algorithm.
    The operator to implement is: 
        UPhi0_Gate*UPhi_Gate
    This operator implements a y Rotation around the input state:
        Ry(Theta)|Phi> where |Phi> is the input State
    Where Theta is given by the square root of the expected value of 
    function f(x) (loaded by R_gate) under a probability density p(x)
    (loaded by P_gate):
        sin(Theta) = sqrt(Expected_p(x)[f(x)])

    Parameters
    ----------

    P_gate : QLM AbstractGate
        Customized AbstractGate for loading probability distribution.
    R_gate : QLM AbstractGate
        Customized AbstractGatel for loading integral of a function f(x)
    
    Returns
    ----------
    
    Q_Gate : AbstractGate
        Customized AbstractGate for Amplitude Amplification Algorithm
    """

    nqbits = R_gate.arity

    @build_gate("Q_Gate", [], arity = nqbits)
    def Q_Gate():
        """
        Function generator for creating an AbstractGate for implementation
        of the Amplification Amplitude Algorithm (Q) 
    
        Returns
        ----------

        Qrout : quantum routine
            Routine for Amplitude Amplification Algorithm 
        """
        Qrout = QRoutine()
        qbits = Qrout.new_wires(nqbits)
        Qrout.apply(UPhi0_Gate(nqbits), qbits)
        UPhi_Gate = Load_UPhi_Gate(P_gate, R_gate)
        Qrout.apply(UPhi_Gate, qbits)
        return Qrout
        #Q = AbstractGate(
        #    "UPhi", 
        #    [],
        #    circuit_generator = Q_generator,
        #    arity = nqbits
        #)
    return Q_Gate()    

