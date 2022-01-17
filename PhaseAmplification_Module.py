"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

from qat.lang.AQASM import AbstractGate, QRoutine, X, Z


def U_Phi0_generator(nqbits):
    """
    Circuit generator for creating gate U_Phi_0 for Phase Amplification Algorithm.
    For a n+1 qbit system where the quantum state can be decomposed by:
    |Psi>_{n+1} =a*|FI_{n}>|1>+b|FI_{n}>|0>. 
    The U_Phi_0 gate implements a reflexion around the state |FI_{n}>|1>. 
    The operator implemented is: I-2|FI_{n}>|0><0|<FI_{n}|
    Inputs:
        *nqbits: int. Number of Qbits of the Abstract Gate
    Output:
        * qrout: quantum routine wiht the circuit implementation
    """
    qrout = QRoutine()
    qbits = qrout.new_wires(nqbits)
    qrout.apply(X, qbits[-1])
    qrout.apply(Z, qbits[-1])
    qrout.apply(X, qbits[-1])
    
    return qrout#-Zeroes
#Definition of the Abstract Gate
U_Phi_0 = AbstractGate("U_Phi_0", [int])
U_Phi_0.set_circuit_generator(U_Phi0_generator)

def D0_generator(nqbits):
    """
    Circuit generator for create an Abstract Gate that implements a Reflexion
    around the state perpendicular to |0>_{n}.
    Implements operator: I-2|0>_{n}{n}<0|=X^{n}c^{n-1}ZX^{n}
    Input:
        * nqbits: number of qbits of the Abstract Gate
    Outputs:
        * qrout: qlm routine that implemnts Operation.
    """
    qrout = QRoutine()
    qbits = qrout.new_wires(nqbits)
    for i in range(nqbits):
        qrout.apply(X, qbits[i])
    #Controlled Z gate by n-1 first qbits
    cZ = 'Z'+ '.ctrl()'*(len(qbits)-1)
    qrout.apply(eval(cZ), qbits[:-1], qbits[-1])
    for i in range(nqbits):
        qrout.apply(X, qbits[i])
    
    return qrout#-Zeroes
D_0 = AbstractGate("D_0", [int])
D_0.set_circuit_generator(D0_generator)

def U_Phi_Gate(nqbits, P_gate, R_gate):
    """
    Create gate U_Phi mandatory for Phase Amplification Algorithm.
    The operator to implement is: I-2|Phi_{n-1}><Phi_{n-1}|. 
    Where the state |Phi_{n-1}> is: |Phi_{n-1}>=R*P*|0_{n+1}>. 
    Where R and P are the gates to load the integral of a function f(x) and 
    the load of a distribution probabilitiy p(x) respectively.
    Inputs:
        * nqbits: int. Number of Qbits of the Gate
        * P_gate: quantum gate for loading probability distribution.
        * R_gate: quantum gate for loading integral of a function f(x)
    Outputs:
        * U_Phi: guantum gate that implements U_Phi gate
    """
    

    def U_Phi_generator(nqbits):
        """
        Circuit generator for the U_Phi_Gate.
        Operation to be implemented: R*P*D_0*P^{+}R^{+}
        Inputs:
            * nqbits: int. Number of Qbits for the circuit
        Outputs:
            * qrout: quantum routine with the circuit implementation
        """
        qrout = QRoutine()
        qbits = qrout.new_wires(nqbits)
        qrout.apply(R_gate.dag(), qbits)
        qrout.apply(P_gate.dag(), qbits[:-1])
        qrout.apply(D_0(nqbits), qbits)
        qrout.apply(P_gate, qbits[:-1])
        qrout.apply(R_gate, qbits)
        return qrout
    U_Phi = AbstractGate("UPhi", [int])
    U_Phi.set_circuit_generator(U_Phi_generator)
    return U_Phi(nqbits)
