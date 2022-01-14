"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

from qat.lang.AQASM import AbstractGate, QRoutine, X, Z


def U_Phi0_generator(nqbits):
    """
    Implementa una reflexion en torno al estado |\Phi_0>: I-2|\Phi_0><Phi_0|
    For a quantum state in the form: |Psi> =a*|Psi1>+b|Psi_0>. This function
    generates a reflection around 
    """
    qrout = QRoutine()
    qbits = qrout.new_wires(nqbits)
    qrout.apply(X, qbits[-1])
    qrout.apply(Z, qbits[-1])
    qrout.apply(X, qbits[-1])
    
    return qrout#-Zeroes
U_Phi0 = AbstractGate("UPhi_0", [int])
U_Phi0.set_circuit_generator(U_Phi0_generator)

def D0_generator(nqbits):
    """
    Circuit generator for create an Abstract Gate that implements a Reflexion
    around the state perpendicular to |0>_{n}. Implements operator: I-2|0>_{n}{n}<0|
    Input:
        * nqbits: number of qbits of the Abstract Gate
    Outputs:
        * qrout: qlm routine that implemnts Operaction.
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
