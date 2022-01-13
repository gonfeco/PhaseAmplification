"""
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""
import numpy as np
import pandas as pd

def TestBins(array, text='Probability'):
    """
    Testing Condition for numpy arrays. The length of the array must be 2^n with n an int.
    Inputs:
    """

    nqbits_ = np.log2(len(array))
    Condition = (nqbits_%2 ==0) or (nqbits_%2 ==1)
    ConditionStr = 'Length of the {} Array must be of dimension 2^n with n an int. In this case is: {}.'.format(text, nqbits_)    
    assert Condition, ConditionStr
    return int(nqbits_)

def LeftConditionalProbability(InitialBins, Probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry Rudolph 2008 papper:
    'Creating superpositions that correspond to efficiently integrable probability distributions'
    http://arXiv.org/abs/quant-ph/0208112v1
    Given a discretized probability and an initial spliting the function splits each initial region in
    2 equally regions and calculates the condicional probabilities for x is located in the left part
    of the new regions when x is located in the region that contains the corresponding left region
    Inputs:
    * InitialBins: int. Number of initial bins for spliting the input probabilities
    * Probability: np.array. Array with the probabilities to be load. 
    InitialBins <= len(Probabilite)
    Outputs:
    * Prob4Left: conditional probabilities of the new InitialBins+1 splits    
    """
    #Initial domain division
    DomainDivisions = 2**(InitialBins)
    
    if DomainDivisions >= len(Probability):
        raise ValueError('The number of Initial Regions (2**InitialBins) must be lower than len(Probability)')
    
    #Original number of bins of the probability distribution
    nbins = len(Probability)
    #Number of Original bins in each one of the bins of Initial domain division 
    BinsByDomainDivision = nbins//DomainDivisions
    #Probability for x located in each one of the bins of Initial domain division
    Prob4DomainDivision = [
        sum(Probability[j*BinsByDomainDivision:j*BinsByDomainDivision+BinsByDomainDivision]) \
        for j in range(DomainDivisions)
    ]
    #Each bin of Initial domain division is splitted in 2 equal parts
    Bins4LeftDomainDivision = nbins//(2**(InitialBins+1))    
    
    #Probability for x located in the left bin of the new splits
    LeftProbs = [
        sum(Probability[j*BinsByDomainDivision:j*BinsByDomainDivision+Bins4LeftDomainDivision])\
        for j in range(DomainDivisions)
    ]    
    #Conditional probability of x located in the left bin when x is located in the 
    #bin of the initial domain division that contains the split
    #Basically this is the f(j) function of the article with j=0,1,2,...2^(i-1)-1 
    #and i the number of qbits of the initial domain division 
    Prob4Left = np.array(LeftProbs)/np.array(Prob4DomainDivision)    
    return Prob4Left

def get_histogram(p, a, b, nbin):
    """
    Given a function p, convert it into a histogram. The function must be positive, the normalization is automatic.
    Note that instead of having an analytical expression, p could just create an arbitrary vector of the right dimensions and positive amplitudes
    so that this procedure could be used to initialize any quantum state with real amplitudes
    
    a    (float)    = lower limit of the interval
    b    (float)    = upper limit of the interval
    p    (function) = function that we want to convert to a probability mass function. It does not have to be normalized but must be positive in the interval
    nbin (int)      = number of bins in the interval
    """
    step = (b-a)/nbin
    centers = np.array([a+step*(i+1/2) for i in range(nbin)]) #Calcula directamente los centros de los bines
    
    prob_n = p(centers)
    assert np.all(prob_n>=0.), 'Probabilities must be positive, so p must be a positive function'
    probs = prob_n/np.sum(prob_n)
    assert np.isclose(np.sum(probs), 1.), 'Probability is not getting normalized properly'
    return centers, probs

def PostProcessResults(Results):
    """
    Post-processing the results of simulation of a quantum circuit
    Input:
        * Results: result object from a simulation of a quantum circuit
    Output:
        * pdf: pandas datasframe. Results of the simulation. There are 3 different columns:
            - States: posible quantum basis states
            - Probability: probabilities of the different states
            - Amplitude: amplitude of the different states
    """
    QP = []
    States = []
    QA = []
    for sample in Results:
        #print("State %s probability %s amplitude %s" % (sample.state, sample.probability, sample.amplitude))
        QP.append(sample.probability)
        States.append(str(sample.state))
        QA.append(sample.amplitude)
    QP = pd.Series(QP, name='Probability')
    States = pd.Series(States, name='States')  
    QA = pd.Series(QA, name='Amplitude') 
    pdf = pd.concat([States, QP, QA], axis=1)
    return pdf     
