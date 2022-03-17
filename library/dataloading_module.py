
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
from qat.lang.AQASM import QRoutine, RY, build_gate
from qat.lang.AQASM.qint import QInt
from AuxiliarFunctions import test_bins, left_conditional_probability


@build_gate("CRBS", [int, int, float], arity=lambda x, y, z: x+1)
def crbs_gate(n_qbits, control_state, theta):
    """
    This functions codify a input control_state using n_qbits qbits and
    apply a controlled Y-Rotation by control_state of theta on one
    aditional qbit.

    Parameters
    ----------

    n_qbits : int
        Number of qbits needed for codify the control_state.
    control_state : int
        State for controlling the of the controlled Rotation.
    theta : float
        Rotation angle (in radians)

    Returns
    ----------

    q_rout : quantum routine.
        Routine for creating a controlled multistate Rotation.

    """
    q_rout = QRoutine()
    #Use of quantum integer types for control qbits
    qcontrol = q_rout.new_wires(n_qbits, QInt)
    #Qbit where rotation should be applied
    qtarget = q_rout.new_wires(1)
    #The control qbits should be equal to the input control_state
    #integer in order to apply the Rotation to the target qbit
    expresion = (qcontrol == control_state)
    #An auxiliar qbit is created for storing the result of the
    #expresion. This qbit will be in state |0> unless the control qbits
    #equals the Integer control_state where state will change to |1>
    with q_rout.compute():
        q_aux = expresion.evaluate()
    #The Rotation on the target qbit will be controlled by the auxiliar
    #qbit which contains the result of the expresion evaluation
    q_rout.apply(RY(theta).ctrl(), q_aux, qtarget)
    #Finally we need to undo the evaluation of the expresion in order to
    #get the original control qbits
    q_rout.uncompute()
    return q_rout

def load_p_gate(probability_array):
    """
    Creates a customized AbstractGate for loading a discretized
    Probability.

    Parameters
    ----------

    probability_array : numpy array
        Numpy array with the discretized probability to load. The number
        of qbits will be log2(len(probability_array)).


    Raises
    ----------
    AssertionError
        if len(probability_array) != 2^n

    Returns
    ----------

    p_gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array
    """

    n_qbits = test_bins(probability_array, 'Probability')

    @build_gate("P", [], arity=n_qbits)
    def p_gate():
        """
        Function generator for the AbstractGate that allows the loading
        of a discretized Probability in a Quantum State.

        Returns
        ----------

        q_rout : Quantum Routine
            Quantum Routine for loading Probability
        """

        q_rout = QRoutine()
        qbits = q_rout.new_wires(n_qbits)
        for i in range(0, n_qbits):
            conditional_probability = left_conditional_probability(
                i, probability_array)
            thetas_list = 2.0*(np.arccos(np.sqrt(conditional_probability)))
            if i == 0:
                #The first qbit is a typical y Rotation
                q_rout.apply(RY(thetas_list[0]), qbits[0])
            else:
                #The different rotations should be applied over the
                #i+1 qbit. Each rotation is controlled by all the
                #posible states formed with i qbits
                for j, theta in enumerate(thetas_list):
                    #Next lines do the following operation:
                    #|j> x Ry(2*\theta_{j})|0>
                    gate = crbs_gate(i, j, theta)
                    q_rout.apply(gate, qbits[:i+1])
        return q_rout
    return p_gate()

def load_f_gate(function_array):
    """
    Creates a customized AbstractGate for loading the integral of a
    discretized function in a Quantum State.

    Parameters
    ----------

    function_array : numpy array
        Numpy array with the discretized function to load.
        The number of qbits will be log2(len(function_array))+1.
        Integral will be load in the last qbit

    Raises
    ----------
    AssertionError
        if len(function_array) != 2^n

    Returns
    ----------

    F_Gate: AbstractGate
        AbstractGate customized for loadin the integral of the function.
    """

    nqbits_ = test_bins(function_array, 'Function')
    #Calculation of the rotation angles
    thetas_list = 2.0*np.arcsin(np.sqrt(function_array))

    @build_gate("F", [], arity=nqbits_+1)
    def f_gate():
        """
        Function generator for creating an AbstractGate that allows
        the loading of the integral of a given discretized function
        array into a Quantum State.

        Returns
        ----------

        q_rout : quantum routine
            Routine for loading the input function as a integral
            on the last qbit.
        """

        q_rout = QRoutine()
        qbits = q_rout.new_wires(nqbits_+1)
        number_of_states = 2**nqbits_
        #Loop over the States
        for i in range(number_of_states):
            #State |i>
            #Generation of a Controlled rotation of theta by state |i>
            q_rout.apply(crbs_gate(nqbits_, i, thetas_list[i]), qbits)
        return q_rout

    return f_gate()

def load_pf(p_gate, f_gate):
    """
    Create complete AbstractGate for applying Operators P and R
    The operator to implement is:
        p_gate*r_gate

    Parameters
    ----------
    p_gate : QLM AbstractGate
        Customized AbstractGate for loading probability distribution.
    f_gate : QLM AbstractGate
        Customized AbstractGatel for loading integral of a function f(x)
    Returns
    ----------
    pr_gate : AbstractGate
        Customized AbstractGate for loading the P and R operators
    """
    nqbits = f_gate.arity
    @build_gate("PF", [], arity=nqbits)
    def load_pf_gate():
        """
        QLM Routine generation.
        """
        q_rout = QRoutine()
        qbits = q_rout.new_wires(nqbits)
        q_rout.apply(p_gate, qbits[:-1])
        q_rout.apply(f_gate, qbits)
        return q_rout
    return load_pf_gate()
