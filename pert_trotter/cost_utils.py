import numpy as np
from scipy.optimize import minimize, Bounds

def countOneRotFerm(Nqubs,NFrags):
    '''
    Returns: number of single qubit rotation gates needed for a single first order Trotter step
    Args:
    Nqubs, width of the circuit, i.e. the number of qubits that encode the Hamiltonian
    NFrags, number of Hamiltonian fragments for a single first order Trotter step
    '''
    dum=2*Nqubs**2 * NFrags-Nqubs
    return dum

def GetTgatUBP(epsilon, Nr, TotErr):
    '''
    Returns: an optimized value for the number of T gates, using numerically exact scalings of Trotter error
    epsilon.
    Args:
    epsilon: numerically exact scaling of trotter error
    Nr, number of single qubit rotation gates
    TotError, target error budget. For chemical accuracy 0.001
    '''

    def objective_function(x, epsilon=epsilon, Nr=Nr, TotErr=TotErr):
        '''
        x[0] is the target trotter error
        x[1] is the target phase estimation error
        '''
        coeff = 0.76 * np.pi * np.sqrt(epsilon) * Nr / (np.sqrt(x[0]) * (TotErr - x[0]))
        fact = 1.15 * np.log2(Nr * np.sqrt(epsilon) / ((TotErr - x[0]-x[1]) * np.sqrt(x[0]))) + 9.2
        return coeff * fact

    # Set the lower and upper bounds for each variable separately
    lb = [1e-8,1e-8]  # Lower bound for x
    ub = [TotErr-1e-8,TotErr-1e-8]  # Upper bound for x

    # Create the bounds object
    bounds = Bounds(lb, ub)

    # Initial guess for the minimization
    initial_guess = [0.4*TotErr,0.1*TotErr]  # Initial guess for x

    # Perform the minimization
    result = minimize(objective_function, initial_guess, bounds=bounds)

    #return result.fun[0]
    return result.fun








#function to find the optimized values
def GetTgatUBalpha(alpha, Nr, epsilon):
    '''
    Returns: an optimized value for the number of T gates, based on Trootter error upper bound \alpha
    Args:
    alpha: Trotter error based on spectral norm of commutators
    Nr:  number of single qubit rotation gates
    epsilon: target accuracy to reach in energy estimation (usually 0.001 Hartrees)

    '''

    def objective_function(x, alpha=alpha, Nr=Nr, epsilon=epsilon):
        coeff = 0.76 * np.pi * alpha * Nr / (x * (epsilon - x))
        fact = 1.15 * np.log2(Nr * alpha / ((1 - epsilon) * x) + 9.2)
        return coeff * fact

    # Set the lower and upper bounds for each variable separately
    lb = [1e-8]  # Lower bound for x
    ub = [epsilon-1e-8]  # Upper bound for x

    # Create the bounds object
    bounds = Bounds(lb, ub)

    # Initial guess for the minimization
    initial_guess = [0.1*epsilon]  # Initial guess for x

    # Perform the minimization
    result = minimize(objective_function, initial_guess, bounds=bounds)

    #return result.fun[0]
    return result.fun, result.x