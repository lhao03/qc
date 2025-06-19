import warnings
from itertools import combinations
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy as sp
from numpy import ndarray, dtype
from numpy.core.numeric import isclose
from opt_einsum import contract
from scipy.optimize import OptimizeResult, minimize

from d_types.fragment_types import GFROFragment, Nums
from min_part.julia_ops import jl_print
from min_part.tensor import tbt2op


def frob_norm(tensor) -> float:
    """Returns the norm as defined by this formula:

    sqrt sum |tensor * tensor|

    Args:
        tensor: an N-rank tensor

    Returns:
        the value of the norm
    """
    return np.sqrt(np.sum(np.abs(tensor * tensor)))


def make_x_matrix(
    thetas: np.ndarray, n: int, diags: Optional[np.ndarray] = None, imag: bool = False
) -> np.ndarray:
    """Makes the X matrix required to define a unitary orbital rotation, where

    U = e^X

    Elements are filled into X starting at a diagonal element at (i, i) and then filling the ith column and ith row.

    So given an N by N matrix, we use n elements in theta for the 1st row and column, then (n-1) elements for the 2nd
    row and column, etc...

    Args:
        thetas: angles required to make the X matrix, need N(N+1)/2 angles
        n: used for the dimension of the X matrix

    Returns:
        an N by N matrix
    """
    expected_num_angles = (n * (n + 1) // 2) - n
    if thetas.size != expected_num_angles:
        raise UserWarning(
            f"Expected {expected_num_angles} angles for a {n} by {n} X matrix, got {thetas.size}."
        )
    if imag:
        if not isinstance(diags, np.ndarray):
            raise UserWarning(
                "Since the X matrix might be imaginary, there might be diagonal elements."
            )
    X = np.zeros((n, n), dtype=np.complex128 if imag else np.float64)
    t = 0
    for x in range(n):
        for y in range(x + 1, n):
            val = thetas[t]
            v_real = val.real
            v_imag = val.imag
            if not isclose(0, v_real) and not isclose(0, v_imag):
                X[x][y] = complex(real=-v_real, imag=v_imag)
                X[y][x] = complex(real=v_real, imag=v_imag)
            elif not isclose(0, v_real) and isclose(0, v_imag):
                X[x][y] = -v_real
                X[y][x] = v_real
            elif isclose(0, v_real) and not isclose(0, v_imag):
                X[x][y] = complex(real=0, imag=v_imag)
                X[y][x] = complex(real=0, imag=v_imag)
            else:
                pass
            t += 1
    if imag:
        for i, d in enumerate(diags):
            if not isclose(d, 0):
                X[i, i] = d
    return X


def make_unitary(thetas: Nums, n: int, imag: bool = False) -> np.ndarray:
    X = make_x_matrix(np.array(thetas), n, imag=imag)
    u = sp.linalg.expm(X)
    u.setflags(write=True)
    num_err = np.finfo(u.dtype).eps
    tol = num_err * 10
    u.real[abs(u.real) < tol] = 0.0
    return u


def make_lambda_matrix(lambdas: np.ndarray, n: int) -> np.ndarray:
    expected_size = n * (n + 1) // 2
    if lambdas.size != expected_size:
        raise UserWarning(
            f"Expected {expected_size} angles for a {n} by {n} lambda matrix, got {lambdas.size}."
        )
    l = np.random.rand(n, n)
    t = 0
    for x in range(n):
        for y in range(x, n):
            l[y][x] = lambdas[t]
            l[x][y] = lambdas[t]
            t += 1
    return l


def make_fr_tensor_from_u(lambdas, u, n) -> np.ndarray:
    """Makes a two-body tensor, defined as sum_{pqrs} sum_{lm} [lambda_{lm} U_lp U_lq U_mr U_ms]

    Checks that the provided unitary matrix is
    1. square
    2. has determinant of 1

    Args:
        lambdas: coefficients for a FR fragment
        u: a unitary matrix used for orbital rotation
        n: shape of the original two-body tensor, where n x n x n x n

    Returns:
        tensor of the FR fragment
    """
    lm = make_lambda_matrix(lambdas, n)
    return contract("lm,lp,lq,mr,ms->pqrs", lm, u, u, u, u)


def make_fr_tensor(lambdas, thetas, n) -> np.ndarray:
    """Makes a two-body tensor, defined as sum_{pqrs} sum_{lm} [lambda_{lm} U_lp U_lq U_mr U_ms]

    Args:
        lambdas: coefficients for a FR fragment
        thetas: angles for the orbital rotation of a FR fragment
        n: shape of the original two-body tensor, where n x n x n x n

    Returns:
        tensor of the FR fragment
    """
    lm = make_lambda_matrix(lambdas, n)
    U = make_unitary(thetas, n)
    return contract("lm,lp,lq,mr,ms->pqrs", lm, U, U, U, U)


def gfro_cost(lambdas, thetas, g_pqrs, n):
    t = n * (n + 1) / 2
    if lambdas.shape[0] != 1 and thetas.shape[0] != t - n:
        raise ValueError(
            "Expanded n * (n + 1) / 2 elements in lambdas and [n * (n + 1) / 2] -n elements in thetas"
        )
    w_pqrs = make_fr_tensor(lambdas, thetas, n)
    diff = g_pqrs - w_pqrs
    output = np.sum(np.abs(diff * diff))
    return output


def gfro_decomp(
    tbt: np.ndarray,
    threshold=1e-6,
    max_iter: int = 10000,
    only_proceed_if_success: bool = False,
    debug: bool = True,
    previous_lambdas: Optional[List[np.ndarray]] = None,
    previous_thetas: Optional[List[np.ndarray]] = None,
) -> list[GFROFragment]:
    """Greedy Full Rank Optimization (GFRO) as described by 'Hamiltonian Decomposition Techniques' by Smik Patel,
    and various Izmaylov group publications.

    Procedure:

    1. Introduce a G tensor. It is initialized to the original two-fermion tensor.
    2. Select an optimal fragment that minimizes the cost function, via non-linear gradient based optimization of lambda and theta.
    3. Update G tensor with chosen fragment, and recalculate L1 norm
    4. Repeat until L1 norm reaches the desired threshold

    Args:
        previous_lambdas:
        previous_thetas:
        debug:
        seed:
        only_proceed_if_success:
        threshold:
        max_iter:
        tbt: two-body operator in np.array form

    Returns:
        list of fragments
    """
    g_tensor = tbt.copy()
    frags: List[GFROFragment] = []
    iter = 0
    n = tbt.shape[0]

    while frob_norm(g_tensor) >= threshold and iter <= max_iter:
        factor = 10 / frob_norm(g_tensor)
        x_dim = n * (n + 1) // 2
        prev_lambda = (
            previous_lambdas[iter]
            if previous_lambdas and iter < len(previous_lambdas)
            else None
        )
        prev_theta = (
            previous_thetas[iter]
            if previous_thetas and iter < len(previous_thetas)
            else None
        )
        greedy_sol = try_find_greedy_fr_frag(
            n,
            threshold,
            g_tensor,
            factor,
            x_dim,
            prev_lambda=prev_lambda,
            prev_theta=prev_theta,
        )
        if only_proceed_if_success:
            greedy_sol = retry_until_success(
                factor,
                g_tensor,
                greedy_sol,
                iter,
                n,
                threshold,
                x_dim,
                prev_lambda=prev_lambda,
                prev_theta=prev_theta,
            )
        lambdas_sol = greedy_sol.x[:x_dim] / factor
        thetas_sol = greedy_sol.x[x_dim:]
        fr_frag_tensor = make_fr_tensor(lambdas_sol, thetas_sol, n)
        frags.append(
            GFROFragment(
                lambdas=lambdas_sol, thetas=thetas_sol, operators=tbt2op(fr_frag_tensor)
            )
        )
        g_tensor -= fr_frag_tensor
        iter += 1
        if debug:
            print(f"Current norm: {frob_norm(g_tensor)}")

    return list(filter(lambda f: len(f.operators.terms) > 0, frags))


def retry_until_success(
    factor,
    g_tensor,
    greedy_sol,
    iter,
    n,
    threshold,
    x_dim,
    prev_lambda,
    prev_theta,
):
    tries = iter
    while not greedy_sol.success:
        warnings.warn(
            UserWarning(
                f"Failed to converge on iteration {iter}, trying again: {str(tries - iter)} try."
            )
        )
        greedy_sol = try_find_greedy_fr_frag(
            n, threshold, g_tensor, factor, x_dim, prev_lambda, prev_theta
        )
        if tries > (100 + iter):
            raise ValueError("Couldn't find good greedy fragment")
        tries += 1
    return greedy_sol


def try_find_greedy_fr_frag(
    n,
    threshold,
    g_tensor,
    factor: float,
    x_dim: int,
    prev_lambda: Optional = None,
    prev_theta: Optional = None,
) -> OptimizeResult:
    x0 = (
        np.concatenate((prev_lambda, prev_theta))
        if isinstance(prev_theta, np.ndarray) and isinstance(prev_lambda, np.ndarray)
        else np.random.uniform(low=-1e-3, high=1e-3, size=(2 * x_dim) - n)
    )
    greedy_sol: OptimizeResult = minimize(
        lambda x0: gfro_cost(
            lambdas=x0[:x_dim], thetas=x0[x_dim:], g_pqrs=factor * g_tensor, n=n
        ),
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": 10000, "disp": False},
        tol=(threshold / n**4) ** 2,
    )
    return greedy_sol


def generate_occupied_spin_orb_permutations(total_spin_orbs: int) -> List[Tuple[int]]:
    possible_spin_orbs = list(range(total_spin_orbs))
    possible_permutations = []
    for i in possible_spin_orbs:
        possible_permutations += list(combinations(possible_spin_orbs, i))
    possible_permutations.append(tuple(possible_spin_orbs))
    return possible_permutations


def gfro_fragment_occ(
    fragment: GFROFragment, num_spin_orbs: int
) -> Tuple[list[Tuple[int]], ndarray[Any, dtype[Any]]]:
    """Given a fragment generated by GFRO, determine the energy of the fragment for all possible electron
    occupation configurations. Assumes `openfermion` spin orbital numbering, where even numbers are spin up, and
    odd numbers are spin down.

    Args:
        fragment: a fragment in the GFRO fragment form
        num_spin_orbs: number of all orbitals (the alpha and beta orbitals count as 2)

    Returns:
        energies of the fragment given a certain occupation of spin orbitals
    """
    lambda_matrix = make_lambda_matrix(fragment.lambdas, num_spin_orbs)
    occupation_combinations = generate_occupied_spin_orb_permutations(num_spin_orbs)
    occ_energies = []
    for occ_comb in occupation_combinations:
        occ_energy = 0
        for l in occ_comb:
            for m in occ_comb:
                occ_energy += lambda_matrix[l][m]
        occ_energies.append(occ_energy)
    return occupation_combinations, np.array(occ_energies)


def extract_thetas(U) -> Tuple[Nums, Nums]:
    """Extracts theta values from a unitary matrix paramertized by real amplitudes.
    Args:
        U: the unitary

    Returns:
        theta values
    """
    X: np.ndarray = sp.linalg.logm(U)
    m = ((U.shape[0] * (U.shape[0] + 1)) // 2) - U.shape[0]
    thetas = np.zeros((m,), dtype=np.complex128)
    u = U.shape[0]
    counter = 0
    for i in range(u - 1):
        for j in range(i + 1, u):
            thetas[counter] = X[j, i]
            counter += 1
    return thetas, X.diagonal()
