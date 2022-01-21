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
#U_Phi_0 = AbstractGate("U_Phi_0", [int])
#U_Phi_0.set_circuit_generator(U_Phi0_generator)
U_Phi_0 = AbstractGate(
    "U_Phi_0",
    [int],
    circuit_generator = U_Phi0_generator,
    arity = lambda x: x
)

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
    c_n_Z = Z.ctrl(nqbits-1)
    #cZ = 'Z'+ '.ctrl()'*(len(qbits)-1)
    #qrout.apply(eval(cZ), qbits[:-1], qbits[-1])
    qrout.apply(c_n_Z, qbits[:-1], qbits[-1])
    for i in range(nqbits):
        qrout.apply(X, qbits[i])
    
    return qrout#-Zeroes
D0_Gate = AbstractGate(
    "D_0",
    [int],
    circuit_generator = D0_generator,
    arity = lambda x: x
)


def U_Phi_Gate(P_gate, R_gate):
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
    
    
    from qat.lang.AQASM import AbstractGate, QRoutine
    #The arity of the R_gate fix the number of qbits for the routine
    nqbits = R_gate.arity
    def U_Phi_generator():
        """
        Circuit generator for the U_Phi_Gate.
        Operation to be implemented: R*P*D_0*P^{+}R^{+}
        Outputs:
            * qrout: quantum routine with the circuit implementation
        """
        
        qrout = QRoutine()
        qbits = qrout.new_wires(nqbits)
        qrout.apply(R_gate.dag(), qbits)
        qrout.apply(P_gate.dag(), qbits[:-1])
        D_0 = D0_Gate(nqbits)
        qrout.apply(D_0, qbits)
        qrout.apply(P_gate, qbits[:-1])
        qrout.apply(R_gate, qbits)
        return qrout
    U_Phi = AbstractGate(
        "UPhi", 
        [],
        circuit_generator = U_Phi_generator,
        arity = nqbits
    )
    return U_Phi()    


def Q_Gate(P_gate, R_gate):
    from qat.lang.AQASM import AbstractGate, QRoutine
    nqbits = R_gate.arity
    def Q_generator():

        qrout = QRoutine()
        qbits = qrout.new_wires(nqbits)
        qrout.apply(U_Phi_0(nqbits), qbits)
        UPhi =U_Phi_Gate(P_gate, R_gate) 
        qrout.apply(UPhi, qbits)
        return qrout
    Q = AbstractGate(
        "UPhi", 
        [],
        circuit_generator = Q_generator,
        arity = nqbits
    )
    return Q()    








