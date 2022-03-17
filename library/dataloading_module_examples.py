"""
This script contains several examples of use of the gates in the
dataloading_module

Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

import numpy as np
from qat.core.console import display
import qat.lang.AQASM as qlm
from qat.qpus import PyLinalg
global_qlmaas = True
try:
    from qlmaas.qpus import LinAlg
except (ImportError, OSError) as exception:
    global_qlmaas = False

from dataloading_module import load_p_gate, load_f_gate, load_pf
from AuxiliarFunctions import get_histogram
from data_extracting import get_results

def get_qpu(qlmass=False):
    """
    Function for selecting solver. User can chose between:
    * LinAlg: for submitting jobs to a QLM server
    * PyLinalg: for simulating jobs using myqlm lineal algebra.

    Parameters
    ----------

    qlmass : bool
        If True  try to use QLM as a Service connection to CESGA QLM
        If False PyLinalg simulator will be used

    Returns
    ----------
    
    lineal_qpu : solver for quantum jobs
    """
    if qlmass:
        if global_qlmaas:
            print('Using: LinAlg')
            linalg_qpu = LinAlg()
        else:
            raise ImportError("""Problem Using QLMaaS.
            Please create config file or use mylm solver""")
    else:
        print('Using PyLinalg')
        linalg_qpu = PyLinalg()
    return linalg_qpu



def Do(n_qbits=6, depth=0, function='DataLoading', qlmass=True):
    """
    Function for testing purpouses. This function is used when the
    script is executed from command line using arguments. It executes
    the three implemented fucntions of this script:
        * load_probability_program
        * load_integral_program
        * expectation_loading_data

    Parameters
    ----------
    n_qbits : int.
        Number of Qbits for the quantum circuit.
    depth : int
        Depth for visualizar the Quantum Circuit
    function : str
        String that indicates which of the before functions should be
        used:
            'P' : load_probability_program
            'I' : load_integral_program
            Otherwise : expectation_loading_data
    """
    def p(x):
        return x*x
    def f(x):
        return np.sin(x)
    #The number of bins
    m_bins = 2**n_qbits
    lower_limit = 0.0
    upper_limit = 1.0
    X, p_x = get_histogram(p, lower_limit, upper_limit, m_bins)
    f_x = f(X)

    linalg_qpu = get_qpu(qlmass)

    if function == 'P':
        print('\t Load Probability')
        p_gate =load_p_gate (p_x)
        results, circuit, q_p, job = get_results(
            p_gate,
            linalg_qpu=linalg_qpu,
            shots=0,
            qubits=list(range(p_gate.arity))
        )
    elif function == 'F':
        print('\t Load Integral')
        f_gate = load_f_gate(f_x)
        q_rout = qlm.QRoutine()
        q_bit = q_rout.new_wires(f_gate.arity)
        for i in range(f_gate.arity-1):
            q_rout.apply(qlm.H, q_bit[i])
        q_rout.apply(f_gate, q_bit)
        results, circuit, q_p, job = get_results(
            q_rout,
            linalg_qpu=linalg_qpu,
            shots=0,
            qubits=[f_gate.arity-1]
        )
    else:
        print('\t Load Data for Expected Value of function')
        p_gate = load_p_gate(p_x)
        f_gate = load_f_gate(f_x)
        pf_gate = load_pf(p_gate, f_gate)
        results, circuit, q_p, job = get_results(
            pf_gate,
            linalg_qpu=linalg_qpu,
            shots=0,
            qubits=[pf_gate.arity-1]
        )

    display(circuit, max_depth=depth)

    print(results.head())
    if function == 'P':
        condition = np.isclose(
            results.sort_values('Int')['Probability'],
            p_x
        ).all()
        print('Probability load data: \n {}'.format(p_x))
        print('Probability Measurements: \n {}'.format(results['Probability']))
        print('This is correct? {}'.format(condition))
    elif function == 'F':
        integral_measurement = results['Probability'][1]*2**(n_qbits)
        print('Integral load data: {}'.format(sum(f_x)))
        print('Integral Measurement: {}'.format(integral_measurement))
        condition = np.isclose(integral_measurement, sum(f_x))
        print('This is correct? {}'.format(condition))
    else:
        integral_measurement = results['Probability'][1]
        print('Integral Measurement: {}'.format(integral_measurement))
        print('Expectation of f(x) for x~p(x): Integral p(x)f(x): {}'.format(sum(p_x*f_x)))
        condition = np.isclose(integral_measurement, sum(p_x*f_x))
        print('This is correct? {}'.format(condition))


if __name__ == '__main__':
    #Working Example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--nqbits',
        type=int,
        help='Number Of Qbits',
        default=6
    )
    parser.add_argument(
        '-depth',
        type=int,
        help='Depth of the Diagram',
        default=0
    )
    parser.add_argument(
        '-t',
        '--type',
        default=None,
        help='Type of Loading: P: Load Probability. F: Load Integral.\
        Otherwise: Load Complete Data'
    )
    parser.add_argument(
        '--qlmass',
        dest='qlmass',
        default=False,
        action='store_true',
        help='For using or not QLM as a Service'
    )
    args = parser.parse_args()
    #print(args)

    Do(
        n_qbits=args.nqbits,
        depth=args.depth,
        function=args.type,
        qlmass=args.qlmass
    )
