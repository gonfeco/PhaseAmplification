"""
This script contains an example of using the implementation of the
Iterative Quantum Phase Estimation algorithm
(module iterative_quantum_pe.py)

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import numpy as np
import pandas as pd

from AuxiliarFunctions import  get_histogram, postprocess_results, test_bins,\
run_job
from QuantumMultiplexors_Module_examples import expectation_loading_data
from PhaseAmplification_Module import load_q_gate
from iterative_quantum_pe import IterativeQuantumPE

def getstaff(InputPDF):
    """
    Function to compute some magnitudes from a result from a iqpe.
    Using the Phi value obtained from iqpae computes angles and Expected
    values.

    Parameters
    ----------

    InputPDF : pandas DataFrame
        Pandas with the results from a complete iqpe solution. The main
        is the Phi column that stores the angles obtained for phase
        estimation (angles must be between 0 and 1)

    Returns
    ----------

    pdf : Pandas DataFrame
        Input pandas DataFrame with some values added:
        Theta_Unitary: is the input angle Phi in radians.
        Theta: is the angle of the rotation for a Groover operator.
        theta_90: is the angle of rotation for a Groover operator
        between [0, pi/2]
        E_p(f): is the expected value for a function f(x) for x following
        a p(x) distribution probability. Basically sin^2(Theta)

    """
    pdf = InputPDF.copy(deep=True)
    #Calculates the eigenvalue for a Unitary Operator
    pdf['Theta_Unitary'] = 2*np.pi*pdf['Phi']
    #A Groover operator implements a 2*Theta rotation. So we calcualte Theta
    pdf['Theta'] = np.pi*pdf['Phi']
    #Restrict Theta values to [0, pi/2]
    pdf['theta_90'] = pdf['Theta']
    pdf['theta_90'].where(
        pdf['theta_90']< 0.5*np.pi,
        np.pi-pdf['theta_90'],
        inplace=True
    )
    #Computes the expected value.
    pdf['E_p(f)'] = np.sin(pdf['Theta'])**2
    return pdf


def Do(n_qbits=6, n_cbits=6, shots=0, QLMASS=True, Save=False):
    """
    Function for testing purpouses. This function is used when the
    script is executed from command line using arguments.

    Parameters
    ----------
    n_qbits : int.
        Number of Qbits for the quantum circuit.
    n_cbits : int
        Number of classical bits for phase estimation
    shots : int
        Number of repetitions of hte circuit
    QLMASS : bool
        For using or not QLM QPU (QLM as a Service)
    Save : bool
        For Saving the final DataFrame with results to a file
    """
    if QLMASS:
        try:
            print('########################################')
            print('#########Connection to QLMaSS###########')
            print('########################################')
            from qat.qlmaas import QLMaaSConnection
            connection = QLMaaSConnection('qlm')
            lin_alg = connection.get_qpu("qat.qpus:LinAlg")
            lineal_qpu = lin_alg()
        except (ImportError, OSError) as e:
            print('Problem: usin PyLinalg')
            from qat.qpus import PyLinalg
            lineal_qpu = PyLinalg()
    else:
        print('User Forces: PyLinalg')
        from qat.qpus import PyLinalg
        lineal_qpu = PyLinalg()

    def p(x):
        return x*x
    def f(x):
        return np.sin(x)
    #The number of bins
    m_bins = 2**n_qbits
    lower_limit = 0.0
    upper_limit = 1.0
    x, p_x = get_histogram(p, lower_limit, upper_limit, m_bins)
    f_x = f(x)

    q_prog, p_gate, r_gate = expectation_loading_data(p_x, f_x)
    q_gate = load_q_gate(p_gate, r_gate)
    zalo_dict = {
        'qpu' : lineal_qpu,
        'cbits_number' : n_cbits,
        'easy': True,
        'shots': shots
    }
    iqpe = IterativeQuantumPE(q_prog, q_gate, **zalo_dict)
    iqpe.iqpe()
    pdf = getstaff(iqpe.results)
    file_name = 'Results_classIQPE_nqbits_{}_ncbits_{}_shots_{}_QLMASS_{}.csv'\
    .format(n_qbits, n_cbits, shots, QLMASS)
    if Save:
        pdf.to_csv(file_name)

if __name__ == '__main__':
    #Working Example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--nqbits',
        type=int,
        help='Number Of Qbits', default=6
    )
    parser.add_argument(
        '-c',
        '--ncbits',
        type=int,
        help='Number Of Classical Bits', default=6
    )
    parser.add_argument(
        '-s',
        '--shots',
        type=int,
        help='Number Of Shots', default=0
    )
    parser.add_argument(
        '--qlmass',
        dest='qlmass',
        default=False,
        action='store_true',
        help='For using or not QLM as a Service'
    )
    parser.add_argument(
        '--save',
        dest='save',
        default=False,
        action='store_true',
        help='For saving or not pandas DataFrame results'
    )
    args = parser.parse_args()
    print(args)
    #Do(n_qbits=args.nqbits, depth=args.depth, function=args.type)
    Do(
        n_qbits=args.nqbits,
        n_cbits=args.ncbits,
        shots=args.shots,
        QLMASS=args.qlmass,
        Save=args.save
    )




