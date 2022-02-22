"""
This module contains necesary functions and classes to implement
Iterative Quantum Amplitude Estimation (IQAE)

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import copy
import numpy as np
import pandas as pd
from qat.lang.AQASM import H, PH
from qat.comm.datamodel.ttypes import OpType

def get_qpu(QLMASS=True):
    if QLMASS:
        try:
            from qat.qlmaas import QLMaaSConnection
            connection = QLMaaSConnection()
            LinAlg = connection.get_qpu("qat.qpus:LinAlg")
            lineal_qpu = LinAlg()
        except (ImportError, OSError) as e:
            print('Problem: usin PyLinalg')
            from qat.qpus import PyLinalg
            lineal_qpu = PyLinalg()
    else:
        print('User Forces: PyLinalg')
        from qat.qpus import PyLinalg
        lineal_qpu = PyLinalg()
    return lineal_qpu

def run_job(result):
    try:
        return result.join()
        #State = PostProcessresults(result.join())
    except AttributeError:
        return result
        #State = PostProcessresults(result)

def im_postprocess(result):
    """
    Post Proccess intermediate measurements from a qlm result.

    Parameters
    ----------

    result : qlm result
        string from a qlm simulation

    Returns
    ----------
    pdf : pandas DataFrame
        contains extracted information from intermediate_measurements
        from a qlm result. Columns:
        BitString : str. String with the bits of the measurements done
            during simulation of the circuit
        BitInt : int. Integer representation of the BitString
        Phi : float. Angle representation of the BitString between [0,1].
        Probability : float. Probability of the measurement of the
            classsical bits.

    """
    bit_list = []
    prob_list = []
    for i, im in enumerate(result.intermediate_measurements):
        if i%2 == 1:
            bit_list.append(str(int(im.cbits[0])))
            prob_list.append(im.probability)

    #Needed order the bits
    bit_list.reverse()
    prob_list.reverse()

    bit_string = ''.join(bit_list)
    bit_int = int(bit_string, 2)
    phi = bit_int/(2**len(bit_list))

    pdf = pd.DataFrame({
        #'Probs': [prob_list],
        'BitString': [bit_string],
        'BitInt': [bit_int],
        'Phi': [phi],
        'Probability': [np.prod(prob_list)],

    })
    return pdf

def get_probability(bit, clasical_bits):
    """
    Calculates the probability of a string of bits based on the
    probabilities for each individual bit

    Parameters
    ----------

    bit : str
        strign of bits that represent an integer number
    clasical_bits : list
        it contains for each position a bolean value and the probability for it
        len(clasical_bits) == len(bit)
        classica_bits[i][0] : bolean value
        classica_bits[i][1] : probability of correspondient bolean value

    Returns
    ----------

    total_probability : float
        Probability of getting input bit having the
        probability configuration of clasical_bits

    """
    p_ = []
    for i, b_ in enumerate(bit):
        #print(i, b_)

        if clasical_bits[i][0] == bool(int(b_)):
            #print('cierto')
            p_.append(clasical_bits[i][1])
        else:
            #print('false')
            p_.append(1.0-clasical_bits[i][1])
        #print(p)
    total_probability = np.prod(p_)
    return total_probability


def step_iqae(q_prog, q_gate, q_aux, c_bits, l):
    """
    Implements a iterative step of the Iterative Phase Estimation (IPE)
    algorithm.

    Parameters
    ----------

    q_prog : QLM program
        QLM Program where the unitary operator will be applied
    q_gate : QLM AbstractGate
        QLM implementation of the unitary operator. We want estimate
        the autovalue theta of this operator
    q_aux : QLM qbit
        auxiliar qbit for IPE. This qbit will be the control
        for application of the unitary operator to the principal qbits
        of the program. Aditionally will be the target qbit for the
        classical bit controlled rotation. This qbit will be reset at
        the end of the step.
    c_bits : list
        list with the classical bits allocated for phase estimation
    l : int
        iteration step of the IPE algorithm

    """
    print('VERSION GONZALO!!')

    q_prog.reset(q_aux)
    #Getting the principal qbits
    q_bits = q_prog.registers[0]
    #First apply a Haddamard Gate to auxiliar qbit
    q_prog.apply(H, q_aux)
    #number of bits for codify phase
    m = len(c_bits)

    #Number of controlled application of the unitary operator by auxiliar
    #qbit over the principal qbits
    unitary_applications = int(2**(m-l-1))
    print('unitary_applications: {}'.format(unitary_applications))
    for i in range(unitary_applications):
        q_prog.apply(q_gate.ctrl(), q_aux, q_bits)
    for j in range(m-l+1, m+1, 1):
        theta = 2**(m-l-j+1)
        print('\t j: {}. theta: {}'.format(j-1, theta))
        q_prog.cc_apply(c_bits[j-1], PH(-(np.pi/2.0)*theta), q_aux)
    print('m: {}. l: {}'.format(m, l))
    q_prog.apply(H, q_aux)
    print(m-l-1)
    q_prog.measure(q_aux, c_bits[m-l-1])
    return q_prog

def step_iqae_easy(q_prog, q_gate, q_aux, c_bits, l):
    """
    Implements a iterative step of the Iterative Phase Estimation (IPE)
    algorithm.

    Parameters
    ----------

    q_prog : QLM program
        QLM Program where the unitary operator will be applied
    q_gate : QLM AbstractGate
        QLM implementation of the unitary operator. We want estimate
        the autovalue theta of this operator
    q_aux : QLM qbit
        auxiliar qbit for IPE. This qbit will be the control
        for application of the unitary operator to the principal qbits
        of the program. Aditionally will be the target qbit for the
        classical bit controlled rotation. This qbit will be reset at
        the end of the step.
    c_bits : list
        list with the classical bits allocated for phase estimation
    l : int
        iteration step of the IPE algorithm

    """

    print('VERSION EASY!!')
    q_prog.reset(q_aux)
    #Getting the principal qbits
    q_bits = q_prog.registers[0]
    #First apply a Haddamard Gate to auxiliar qbit
    q_prog.apply(H, q_aux)
    #number of bits for codify phase
    m = len(c_bits)

    #Number of controlled application of the unitary operator by auxiliar
    #qbit over the principal qbits
    unitary_applications = int(2**(m-l-1))
    print('unitary_applications: {}'.format(unitary_applications))
    for i in range(unitary_applications):
        q_prog.apply(q_gate.ctrl(), q_aux, q_bits)

    for j in range(l):
        theta = 1.0/(2**(l-j-1))
        print('\t j: {}. theta: {}'.format(j, theta))
        q_prog.cc_apply(c_bits[j], PH(-(np.pi/2.0)*theta), q_aux)

    #for j in range(m-l+1, m+1, 1):
    #    theta = 2**(m-l-j+1)
    #    print('\t j: {}. theta: {}'.format(j-1, theta))
    #    q_prog.cc_apply(c_bits[j-1], PH(-(np.pi/2.0)*theta), q_aux)

    print('m: {}. l: {}'.format(m, l))
    q_prog.apply(H, q_aux)
    #print(m-l-1)
    #q_prog.measure(q_aux, c_bits[m-l-1])
    q_prog.measure(q_aux, c_bits[l])
    return q_prog


class IterativeQuantumAE:

    def __init__(self, q_prog, q_gate, **kwargs):
        #Initial Quatum Program. For restarting purpouses
        self.init_q_prog = q_prog
        #Quantum Gate to apply to quantum program
        self.q_gate = q_gate
        #Setting attributes
        #Number Of classical bits for estimating phase
        self.cbits_number_ = kwargs.get('cbits_number', 8)
        #Set the QPU to use
        self.lineal_qpu = kwargs.get('qpu', get_qpu())
        self.shots = kwargs.get('shots', 0)
        self.restart()
        self.easy = kwargs.get('easy', False)

    def restart(self):
        self.q_prog = None
        self.q_aux = None
        self.c_bits = None
        self.circuit = None
        self.meas_gates = None
        self.job = None
        self.job_result = None
        #self.cbit_solution = None
        #self.measured_solution_bits = None
        #self.measured_solution_int = None
        #self.measured_phi = None
        #self.measured_probability = None
        self.results = None



    @property
    def cbits_number(self):
        return self.cbits_number_

    @cbits_number.setter
    def cbits_number(self, value):
        print('The number of classical bits for phase estimation will be:'\
            '{}'.format(value))
        self.cbits_number_ = value
        #We update the allocate classical bits each time we change cbits_number

    def init_iqae(self):#, number_of_cbits=None):
        self.restart()
        self.q_prog = copy.deepcopy(self.init_q_prog)
        self.q_aux = self.q_prog.qalloc(1)
        self.c_bits = self.q_prog.calloc(self.cbits_number)

    def apply_iqae(self):
        for l in range(len(self.c_bits)):
            if self.easy:
                step_iqae_easy(self.q_prog, self.q_gate, self.q_aux, self.c_bits, l)
            else:
                step_iqae(self.q_prog, self.q_gate, self.q_aux, self.c_bits, l)

    def get_circuit(self):
        self.circuit = self.q_prog.to_circ(submatrices_only=True)
        self.meas_gates = [i for i, o in enumerate(self.circuit.ops)\
            if o.type == OpType.MEASURE]

    def get_job(self):
        self.job = self.circuit.to_job(
            qubits=self.q_aux,
            nbshots=self.shots,
            aggregate_data=False
        )

    def get_job_result(self, qpu=None):
        if qpu is not None:
            self.lineal_qpu = qpu
        self.job_result = run_job(self.lineal_qpu.submit(self.job))
    
    def iqae(self, number_of_cbits=None, shots=None):

        if number_of_cbits is not None:
            self.cbits_number = number_of_cbits
        if shots is not None:
            self.shots = shots

        self.init_iqae()
        self.apply_iqae()
        self.get_circuit()
        self.get_job()
        self.get_job_result()
        self.get_classicalbits()


    def get_classicalbits(self):
        """
        This method gets the classical bits and their probabilities from
        the job_result property
        """

        list_of_results = [im_postprocess(r) for r in self.job_result]
        self.results = pd.concat(list_of_results)
        self.results.reset_index(drop=True, inplace=True)

        #if len(self.job_result) == 1:
        #    results_0 = self.job_result[0]
        #    step_im = results_0.intermediate_measurements
        #    self.cbit_solution = [
        #        [im.cbits[0], im.probability] for im in step_im\
        #        if im.gate_pos in self.meas_gates
        #    ]
        #    if len(self.cbit_solution) == 0:
        #        #For myqlm
        #        self.cbit_solution =[
        #            [bool(im.cbits[0]), im.probability] for i, im in\
        #            enumerate(step_im) if i%2 == 1
        #        ]
        #else:
        #    raise ValueError('Problem with the measured qbit!!')

        #self.cbit_solution.reverse()

    #def get_solutions(self):
    #    """
    #    DISABLE Method. Not Correct
    #    This method creates de pandas DataFrame with the final solution.
    #    Pandas DataFrame will have following columns:
    #        Int : Posible integers from cbit_solution
    #        Bol : bolean string of the posible integers from cbit_solution
    #        Probability : Probability of each posible integer
    #        theta : posible estimation angel based on each posible integer
    #    """


    #    #posible integers that can be represented by len(self.cbit_solution)
    #    int_list = list(range(2**len(self.cbit_solution)))
    #    #string bolean representation of each posible integer
    #    bits_list = [format(i, "b").zfill(len(self.cbit_solution))\
    #        for i in range(2**len(self.cbit_solution))]
    #    #probabilit of each integer based on probabilities of self.cbit_solution
    #    prob_list = []
    #    for b in bits_list:
    #        prob_list.append(get_probability(b, self.cbit_solution))
    #    self.results = pd.DataFrame({
    #        'Int' : int_list,
    #        'Bol' : bits_list,
    #        'Probability': prob_list
    #    })
    #    #We will give phi instead theta. If you want theta: theta=2*pi*phi
    #    #For expectation value of a function remeber that Q operator
    #    #is a rotation of 2*theta. So if you want theta: theta=pi*phi
    #    print('In final solution Phi instead of theta is given')
    #    print('For theta yo need: theta = 2*np.pi*phi')
    #    print('For expectation value of a function Q operator')
    #    print('implements a rotation of 2*theta. In this case do:')
    #    print('theta = np.pi*phi')
    #    #2*theta so 2*theta = 2*pi*thetas
    #    n_total = 2**len(self.cbit_solution)
    #    self.results['phi'] = self.results['Int']/n_total

    #def meas_solution(self):
    #def get_solutions(self):
    #    self.measured_solution_bits = ''.join([str(int(i[0])) for i in\
    #        self.cbit_solution])
    #    self.measured_solution_int = int(self.measured_solution_bits, 2)
    #    n_total = 2**len(self.cbit_solution)
    #    self.measured_phi = self.measured_solution_int/n_total
    #    self.measured_probability = np.prod([i[1] for i in self.cbit_solution])
    #    self.results = pd.DataFrame({
    #        'BitString': [self.measured_solution_bits],
    #        'BitInt': [self.measured_solution_int],
    #        'Phi': [self.measured_phi],
    #        'Probability': [self.measured_probability]
    #    })


