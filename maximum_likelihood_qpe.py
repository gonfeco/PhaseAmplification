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

def apply_gate(q_prog, q_gate, m_k, lineal_qpu, nbshots=0):
    """
    Apply the self.q_gate to the circuit a input number of times
    This method creates a quantum program that applies the
    Q operator n_ops times on the circuit where the probability
    and the function were loaded

    Parameters
    ----------
    q_prog : QLM quantum program
        QLM quantum program with initial configuration
    q_gate : QLM gate
        QLM gate with the Groover-like operator to be applied
    m_k : int
        number of times to apply the q_gate to the q_prog
    lineal_qpu : QLM solver
        QLM solver for submitting a QLM job 
    nbshots : int
        number of shots to perform by the simulator or the QPU
    Returns
    ----------

    pdf : pandas dataframe
        results of the measurement of the last qbit
    circuit : QLM circuit object
        circuit object generated for the quantum program
    job : QLM job object
        job object generated for the quantum circuit
    """
    prog_q = copy.deepcopy(q_prog)
    q_bits = prog_q.registers[0]
    for _ in range(m_k):
        prog_q.apply(q_gate, q_bits)
    circuit = prog_q.to_circ(submatrices_only=True)
    job = circuit.to_job(qubits=[len(q_bits)-1], nbshots=nbshots)
    result = run_job(lineal_qpu.submit(job))
    pdf_ = postprocess_results(result)
    #Change the result presentation
    pdf = get_probabilities(pdf_)
    #Added the number of operations
    pdf['m_k'] = m_k
    return pdf, circuit, job

def get_probabilities(InputPDF):
    pdf = InputPDF.copy(deep=True)
    columns = ['Probability_{}'.format(i) for i in pdf['States']]
    output_pdf = pd.DataFrame(
        pdf['Probability'].values.reshape(1, len(pdf)),
        columns = columns
    )
    return output_pdf

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

class MaximumLikelihoodQPE:

    def __init__(self, q_prog, q_gate, **kwargs):
        """

        Method for initializing the class
    
        Parameters
        ----------
        
        q_prog : QLM quantum program
            Quantum program where the Groover-like operator will be applied
        q_gate : QLM gate
            QLM gate that implements the Groover-like operator
        kwars : dictionary
            dictionary that allows the configuration of the ML-QPE algorithm:
            Implemented keys:
            list_of_mks : list
                python list with the different m_ks for executing the algortihm
            qpu : QLM solver
                solver for simulating the resulting circutis
            delta : float 
                For avoiding problems when calculating the domain for theta
            default_nbshots : int
                default number of measurements for computing freqcuencies
                when nbshots for quantum job is 0
            iterations : int
                number of iterations of the optimizer
            display : bool
                for displaying additional information in the optimization step
            nbshots : int
                number of shots for quantum job. If 0 exact probabilities
                will be computed. 
        """
        #Quatum Program
        self.q_prog = q_prog
        #Quantum Gate to apply to quantum program
        self.q_gate = q_gate
        #A complete list of m_k
        self.list_of_mks = kwargs.get('list_of_mks', None)
        if self.list_of_mks is not None:
            self.list_of_mks = range(10) 
            print('list_of_mks not provide.list_of_mks: {} will be used'.format(
                self.list_of_mks))
        #Set the QPU to use
        self.lineal_qpu = kwargs.get('qpu', get_qpu())
        ##delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get('delta', 1.0e-5)
        #This is the default number of shots used for computing
        #the freqcuencies of the results when the computed probabilities
        #instead of freqcuencies are provided (nbshots = 0 when qlm job
        #is created)
        self.default_nbshots = kwargs.get('default_nbshots', 100)
        #number of iterations for optimization of Likelihood
        self.iterations = kwargs.get('iterations', 100)
        #For displaying extra info for optimization proccess
        self.disp = kwargs.get('disp', True)
        #If 0 we compute the exact probabilities
        self.nbshots = kwargs.get('nbshots', 0)
        #Setting attributes
        self.restart()

    def restart(self):
        self.pdf_mks = None
        self.list_of_circuits = None
        self.list_of_jobs = None
        self.theta = None


    def apply_gate(self, m_k, nbshots):
        """
        This method apply the self.q_gate to the self.q_prog an input
        number of times, creates the correspondient circuit and job,
        submit the job an get the results for an input number of shots.

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit
        nbshots : int
            number of shots to perform by the simulator or the QPU

        Returns
        ----------

        pdf : pandas dataframe
            results of the measurement of the last qbit
        circuit : QLM circuit object
            circuit object generated for the quantum program
        job : QLM job object
        """

        pdf, circuit, job = apply_gate(
            self.q_prog,
            self.q_gate,
            m_k,
            self.lineal_qpu,
            nbshots=nbshots
        )

        #For Maximum Likelihood we need the number of |1> measurements,
        #h_k, and the number of total measurements n_k. QLM gives
        #probabilities so we need to change them to counts
        if nbshots == 0:
            #In this case QLM give simulated probabilities so we fixed
            #to self.default_nbshots
            n_k = self.default_nbshots
        else:
            #In this case we use the proper number of total measurements
            n_k = nbshots
        pdf['h_k'] = round(
            pdf['Probability_|1>']*n_k, 0
        ).astype(int)
        pdf['n_k'] = n_k
        return pdf, circuit, job

    def run(self, list_of_mks=None, nbshots=None):
        """
        This method is the core of the Maximum Likelihood Amplitude
        Estimation. It runs several quantum circuits each one increasing
        the number of self.q_gate applied to the the initial self.q_prog

        """

        if list_of_mks is not None:
            self.list_of_mks = list_of_mks
        if nbshots is not None:
            self.nbshots = nbshots
        #Clean the list in each run
        self.list_of_circuits = []
        pdf_list = []
        for m_k in self.list_of_mks:
            step_circuit, step_pdf, step_job = self.apply_gate(m_k, self.nbshots)
            self.list_of_circuits.append(step_circuit)
            self.list_of_jobs.append(step_job) 
            pdf_list.append(step_pdf)
        self.pdf_mks = pd.concat(pdf_list)
        self.pdf_mks.reset_index(drop=True, inplace=True)
        self.theta = self.launch_optimizer(self.pdf_mks)


    def launch_likelihood(self, pdf_input, N=100):
        """
        This method calculates the Likelihood for theta between [0, pi/2]
        for an input pandas DataFrame. 

        Parameters
        ----------

        pdf_input: pandas DataFrame
            The DataFrame should have following columns:
            m_k: number of times q_gate was applied
            h_k: number of times the state |1> was measured
            n_k: number of total measuremnts

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
        pdf = pdf_input.copy(deep=True)
        if pdf is None:
            print(
                """
                Can not calculate Likelihood because pdf_input is empty.
                Please provide a valida DataFrame.
                """)
            return None
        theta = np.linspace(0+self.delta, 0.5*np.pi-self.delta, N)
        m_k = pdf['m_k']
        h_k = pdf['h_k']
        n_k = pdf['n_k']
        l_k = np.array([likelihood(t, m_k, h_k, n_k) for t in theta])
        y_ = pd.DataFrame({'theta': theta, 'l_k': l_k})
        return y_

    def launch_optimizer(self, results):
        """
        This functions execute a brute force optimization of the
        likelihood function for an input results pdf.

        Parameters
        ----------

        results : pandas DataFrame
            DataFrame with the results from ml-qpe procedure.
            Mandatory columns:
            m_k : number of times Groover like operator was applied 
            h_k : number of measures of the state |1>
            n_k : number of measurements done


        Returns
        ----------

        optimum_theta : float
            theta  that minimize likelihood
        """

        theta_domain = (0+self.delta, 0.5*np.pi-self.delta)
        optimizer = so.brute(
            likelihood,
            [theta_domain],
            (results['m_k'], results['h_k'], results['n_k']),
            self.iterations,
            disp=self.disp
        )
        optimum_theta = optimizer[0] 
        return optimum_theta











        
