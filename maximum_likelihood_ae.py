"""
This module contains necesary functions and classes to implement
Maximu Likelihood Amplitude Estimation based on the papper:

    Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N.
    Amplitude estimation without phase estimation
    Quantum Information Processing, 19(2), 2020
    arXiv: quant-ph/1904.10246v2

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import copy
import numpy as np
import pandas as pd
import scipy.optimize as so

from AuxiliarFunctions import run_job, postprocess_results

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

def likelihood(theta, m_k, h_k, n_k):
    """
    Calculates Likelihood from Suzuki papper. For h_k positive events
    of n_k total events this function calculates the probability of
    this taking into account that the probability of a positive
    event is given by theta and by m_k
    The idea is use this function to minimize it for this reason it gives
    minus Likelihood

    Parameters
    ----------

    theta : float
        Angle (radians) for calculating the probability of measure a
        positive event.
    m_k : pandas Series
        For MLQAE this a pandas Series where each row is the number of
        times the operator Q was applied.
        We needed for calculating the probability of a positive event
        for eack posible m_k value: sin((2*m_k+1)theta)**2.
    h_k : pandas Series
        Pandas Series where each row is the number of positive events
        measured for each m_k
    n_k : pandas Series
        Pandas Series where each row is the number of total events
        measured for each m_k

    Returns
    ----------

    float
        Gives the -Likelihood of the inputs

    """
    theta_ = (2*m_k+1)*theta
    first_term = 2*h_k*np.log(np.abs(np.sin(theta_)))
    second_term = 2*(n_k-h_k)*np.log(np.abs(np.cos(theta_)))
    l_k = first_term + second_term
    return -np.sum(l_k)


def get_sub_pdf(input_pdf, state):
    """
    From an input pandas df extract the probability of a given state

    Parameters
    ----------

    input_pdf : pandas dataframe
        pandas dataframe with at least 2 columns called:
        States and Probability. Ideally it will have two rows with the
        results of 1 qbit measurement
    state : str
        name of the state we want to extract. Ideally it wil be |0> or |1>

    Returns
    ----------

    phi : pandas dataframe
        pandas with the probability of the input state

    """
    pdf = input_pdf.copy(deep=True)
    phi = pdf[pdf['States'] == state]
    if len(phi) == 0:
        #state does not exist. State was not measure.
        #We fix to cero probability
        phi = pd.DataFrame([0.0], columns=['Probability_{}'.format(state)])
    else:
        drop_columns = [c for c in pdf.columns if c != 'Probability']
        phi.reset_index(drop=True, inplace=True)
        phi.drop(columns=drop_columns, inplace=True)
        phi.rename(
            columns={'Probability': 'Probability_{}'.format(state)},
            inplace=True
        )
    return phi




class MaximumLikelihoodAE:

    def __init__(self, q_prog, q_gate, **kwargs):
        #Quatum Program
        self.q_prog = q_prog
        #Quantum Gate to apply to quantum program
        self.q_gate = q_gate

        #Setting attributes
        #A complete list of m_k
        self.list_of_mks = kwargs.get('list_of_mks', None)
        #We create a list of ks by iterating from 0 to K
        self._K = kwargs.get('K', None)
        if (self.list_of_mks is not None) and (self.K is not None):
            print("Two different strategies for creating the list of m_ks were given:"\
            "K and list_of_mks. list_of_mks will be used")
        else:
            if self.K is not None:
                self.K = self._K
            else:
                print('Setting K to {}'.format(10))
                self.K = 10
        self.nbshots = kwargs.get('nbshots', 0)
        #Set the QPU to use
        self.lineal_qpu = kwargs.get('qpu', get_qpu())
        #Number of Measurements for the last Qbit
        #If 0 we compute the exact probabilities
        self.nbshots = kwargs.get('nbshots', 0)
        #delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get('delta', 1.0e-9)
        #number of iterations for optimization of Likelihood
        self.iterations = kwargs.get('iterations', 100)
        #For displaying extra info for optimization proccess
        self.disp = kwargs.get('disp', True)

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        #Allows updating list_of_mks if new K is given
        self._K = value
        print('Updating list_of_mks using {}'.format(self._K))
        self.list_of_mks = range(self._K)

    def run(self, list_of_mks=None):
        """
        This method is the core of the Maximum Likelihood Amplitude
        Estimation. It runs several quantum circuits each one increasing
        the number of self.q_gate applied to the the initial self.q_prog

        """

        if list_of_mks is not None:
            self.list_of_mks = list_of_mks
        #Clean the list in each run
        self.list_of_circuits = []
        pdf_list = []
        for m_k in list_of_mks:
            step_circuit, step_pdf = self.apply_gate(m_k)
            self.list_of_circuits.append(step_circuit)
            pdf_list.append(step_pdf)
        self.p_mks = pd.concat(pdf_list)
        self.p_mks.reset_index(drop=True, inplace=True)

        #For Maximum Likelihood we need the number of |1> measurements,
        #h_k, and the number of total measurements n_k. QLM gives
        #probabilities so we need to change them to counts
        if self.nbshots == 0:
            #In this case QLM give simulated probabilities so we fixed
            #by giving an arbitrary value
            n_k = 100
        else:
            #In this case we use the proper number of total measurements
            n_k = self.nbshots
        self.p_mks['h_k'] = round(
            self.p_mks['Probability_|1>']*n_k, 0
        ).astype(int)
        self.p_mks['n_k'] = n_k

    def apply_gate(self, n_ops=1):
        """
        Apply the self.q_gate to the circuit a input number of times
        This method creates a quantum program that applies the
        Q operator n_ops times on the circuit where the probability
        and the function were loaded

        Parameters
        ----------
        n_ops : int
            number of times to apply the self.q_gate to the quantum circuit

        Returns
        ----------

        circuit : QLM circuit object
            circuit object generated for the quantum program
        pdf : pandas dataframe
            results of the measurement of the last qbit
        """
        prog_q = copy.deepcopy(self.q_prog)
        q_bits = prog_q.registers[0]
        for _ in range(n_ops):
            prog_q.apply(self.q_gate, q_bits)
        circuit = prog_q.to_circ(submatrices_only=True)
        job = circuit.to_job(qubits=[len(q_bits)-1], nbshots=self.nbshots)
        result = run_job(self.lineal_qpu.submit(job))
        pdf_ = postprocess_results(result)
        pdf = pd.concat(
            [get_sub_pdf(pdf_, '|0>'), get_sub_pdf(pdf_, '|1>')],
            axis=1
        )
        pdf['m_k'] = n_ops
        return circuit, pdf

    def launch_likelihood(self, N=100):
        """
        This method calculates the Likelihood for theta between [0, pi/2]
        for the p_mks atribute. The p_mks is a pandas Dataframe that
        should have following columns:
            m_k: number of times q_gate was applied
            h_k: number of times the state |1> was measured
            n_k: number of total measuremnts

        Parameters
        ----------

        N : int
            number of division for the theta interval

        Returns
        ----------

        y : pandas DataFrame
            Dataframe with the likelihood for the p_mks atribute. It has
            2 columns:
            theta : posible valores of the angle
            l_k : likelihood for each theta

        """
        if self.p_mks is None:
            print(
                """
                Can not calculate Likelihood because p_mks is empty.
                Please provide a valida DataFrame or execute run() method
                """)
            return None
        theta = np.linspace(0+self.delta, 0.5*np.pi-self.delta, N)
        m_k = self.p_mks['m_k']
        h_k = self.p_mks['h_k']
        n_k = self.p_mks['n_k']
        l_k = np.array([likelihood(t, m_k, h_k, n_k) for t in theta])
        y_ = pd.DataFrame({'theta': theta, 'l_k': l_k})
        return y_

    def optimize(self):
        """
        This methods estimate the optimal theta that maximizes the
        Likelihood of the obtaining measurments of the Suzuki algorithm.
        Uses the brute method of the scipy optimization library.

        """
        if self.p_mks is None:
            print(
                """
                Can not calculate Likelihood because p_mks is empty.
                Please provide a valid DataFrame or execute run() method
                """)
            return None
        optimizer = so.brute(
            likelihood,
            [(0+self.delta, 0.5*np.pi-self.delta)],
            (self.p_mks['m_k'], self.p_mks['h_k'], self.p_mks['n_k']),
            self.iterations,
            disp=self.disp
        )
        self.theta = optimizer[0]
        return self.theta
