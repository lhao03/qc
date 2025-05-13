import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle


def countOneRotFerm(Nqubs, NFrags, ord=1):
    """
    Returns: number of single qubit rotation gates needed for a single Trotter step of Fermionic fragments
    Args:
    Nqubs, width of the circuit, i.e. the number of qubits that encode the Hamiltonian
    NFrags, number of Hamiltonian fragments
    ord, order of the Trotter approximation. 1 for first order, 2 for second order.
    """
    if ord == 1:
        dum = 2 * Nqubs**2 * NFrags - Nqubs
    elif ord == 2:
        dum = (
            Nqubs**2 * (4 * NFrags - 3) - Nqubs
        )  # This follows from the fact that 2nd order Trotter has 2Nfrags -1 exponentials and of them, 2 are one-body operators.
    return dum


def count_T_gates(eps_vals, eps, Tot_Err, Nr, scale):
    """Function to compute N_T based on the given parameters. eps_vals are the parameters we are optimizng. eps can be perturbative Trotter error or the operator norm Trotter error."""
    if len(eps_vals) == 3:
        eps_HT, eps_TS, eps_PE = eps_vals
    elif len(eps_vals) == 2:
        eps_TS, eps_PE = eps_vals
        eps_HT = Tot_Err - eps_TS - eps_PE

    # Ensure positivity
    if eps_HT < 0 or eps_TS < 0 or eps_PE < 0:
        return np.inf
    elif sum(eps_vals) > Tot_Err:
        return np.inf

    # Compute N_T based on the given formula
    coeff = (0.76 * np.pi * Nr * np.sqrt(eps)) / (np.sqrt(eps_TS) * eps_PE)
    fact = 1.15 * np.log2((Nr * np.sqrt(eps)) / (eps_HT * np.sqrt(eps_TS))) + 9.2
    cost = coeff * fact / scale
    # print ("{:.4e}".format(cost), '\t', sum(eps_vals))
    return cost


def constraint(eps_vals, Tot_Err):
    """Constraint function to ensure sum of epsilons is within limit"""
    return Tot_Err - sum(eps_vals)


def optimize_T_gates(x0, eps, Tot_Err, Nr, scale, use_constraints=True):
    """Minimization of N_T subject to constraints"""

    if use_constraints == True:
        # Bounds: each epsilon should be non-negative
        bounds = [(0, Tot_Err)] * 3
        # Constraint: sum of epsilons <= Tot_Err
        constraints = {"type": "ineq", "fun": constraint, "args": (Tot_Err,)}
        # Perform minimization
        result = minimize(
            count_T_gates,
            x0,
            args=(eps, Tot_Err, Nr, scale),
            bounds=bounds,
            constraints=constraints,
            method="cobyla",
            options={"maxiter": 1000},
        )

        # nonlinear_constraint = NonlinearConstraint(lambda x: constraint(x, Tot_Err), 0, np.inf)

        # result = differential_evolution(func=lambda x: count_T_gates(x, eps, Tot_Err, Nr), bounds=bounds, constraints=(nonlinear_constraint,),
        #                                 strategy='rand1bin', maxiter=1000, polish=False, disp=False)
    else:
        # Bounds: each epsilon should be non-negative
        lb = [1e-8, 1e-8]  # Lower bound for x
        ub = [Tot_Err - 1e-8, Tot_Err - 1e-8]  # Upper bound for x

        # Create the bounds object
        bounds = Bounds(lb, ub)

        x0_new = [x0[1], x0[2]]
        # Perform minimization
        result = minimize(
            count_T_gates, x0_new, args=(eps, Tot_Err, Nr, scale), bounds=bounds
        )

    return result.x, result.fun  # Returns optimized epsilons and minimum N_T


def explore_params(denom, Tot_Err, err_est, Nr, scale, use_constraints=True):
    "Since the optimization of T gates is highly sensitive to the initial parameter values, we explore different initial parameters on a grid to get a better T gate estimate"
    global_min_Nt = np.inf
    for i in range(denom):
        for j in range(denom):
            for k in range(denom):
                if i + j + k + 3 < denom:
                    x0 = [
                        (i + 1) * Tot_Err / denom,
                        (j + 1) * Tot_Err / denom,
                        (k + 1) * Tot_Err / denom,
                    ]
                    optimal_errs, min_Nt = optimize_T_gates(
                        x0, err_est, Tot_Err, Nr, scale, use_constraints=use_constraints
                    )
                    if min_Nt * scale < global_min_Nt:
                        global_min_Nt = min_Nt * scale
                        global_optimal_errs = optimal_errs
    return global_optimal_errs, global_min_Nt
