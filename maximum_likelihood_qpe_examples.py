"""
This script contains an example of using the implementation of the
Maximum Likelihood Phase Estimation algorithm
(module: maximum_likelihood_qpe.py)

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import numpy as np
import pandas as pd

from AuxiliarFunctions import  get_histogram, postprocess_results, test_bins,\
run_job
from QuantumMultiplexors_Module_examples import expectation_loading_data
from PhaseAmplification_Module import load_q_gate
from  maximum_likelihood_qpe import MaximumLikelihoodQPE, likelihood

def Do(n_qbits=6, shots=0, n_thetas=100, QLMASS=True, Save=False,\
max_number_ks=10):
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
        'nbshots': shots,
        'list_of_mks': range(max_number_ks)
    }
    ml_qpe = MaximumLikelihoodQPE(q_prog, q_gate, **zalo_dict)
    ml_qpe.run()
    print(ml_qpe.pdf_mks)
    theoric_theta = np.arcsin(sum(f_x*p_x)**0.5)
    ml_qpe_theta = ml_qpe.theta
    print('theoric_theta: {}'.format(theoric_theta))
    print('ml_qpe_theta: {}'.format(ml_qpe_theta))
    E_f_p = np.sin(ml_qpe_theta)**2
    lk_ = likelihood(
        ml_qpe_theta,
        ml_qpe.pdf_mks['m_k'],
        ml_qpe.pdf_mks['h_k'],
        ml_qpe.pdf_mks['n_k']
    )
    pdf = pd.DataFrame({
        'TheoricTheta': [theoric_theta],
        'Theta' : [ml_qpe_theta],
        'Likelihood' : [lk_],
        'E_p(f)': [E_f_p],
    })
    print(pdf)
    pdf_like = ml_qpe.launch_likelihood(ml_qpe.pdf_mks, n_thetas)
    if Save:
        file_name = 'Thetas_vs_Likelihoods_domain_{}_nqbits_{}_shots_{}_QLMASS_{}.csv'\
        .format(n_thetas, n_qbits, shots, QLMASS)
        pdf_like.to_csv(file_name)
        file_name = 'Results_mks_{}_nqbits_{}_shots_{}_QLMASS_{}.csv'\
        .format(max_number_ks, n_qbits, shots, QLMASS)
        ml_qpe.pdf_mks.to_csv(file_name)
    #print(pdf)

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
        '-s',
        '--shots',
        type=int,
        help='Number Of Shots', default=0
    )
    parser.add_argument(
        '-m_ks',
        type=int,
        help='Max Number of Groover-like operator applications', default=5
    )
    parser.add_argument(
        '-nt',
        '--thetas',
        type=int,
        help='Number Of Thetas for plotting likelihood', default=0
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
        shots=args.shots,
        max_number_ks=args.m_ks,
        n_thetas=args.thetas,
        QLMASS=args.qlmass,
        Save=args.save
    )




