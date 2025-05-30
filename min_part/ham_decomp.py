import warnings
from typing import Any, List

import numpy as np
import scipy as sp
import scipy.linalg
from openfermion import FermionOperator
from opt_einsum import contract
from scipy.optimize import OptimizeResult, minimize

from min_part.reorder import reorder_operators_for_lr
from min_part.tensor import get_n_body_tensor
from min_part.tensor_utils import tbt2op
from min_part.typing import GFROFragment


def make_supermatrix(tbt: np.ndarray) -> np.ndarray:
    """Convert the two-body tensor, a rank 4 tensor,  into a rank 2 tensor, or a matrix.

    Args:
        tbt: two-body tensor with the shape (N x N x N x N), where `tbt[p][q][r][s]` = `V[p][q][r][s]`

    Returns:
        the two-body tensor with the dimensions (N^2 x N^2), where `tbt[pq][rs] = `V[p][q][r][s]`
    """
    n = tbt.shape[0]
    n_sq = n**2
    supermatrix = tbt.reshape((n_sq, n_sq))
    return supermatrix


def four_tensor_to_two_tensor_indices(p, q, r, s, n) -> tuple[Any, Any]:
    """Given an indexing into a rank four tensor: (q, p, r, s), get the rank two tensor indexing: (pq, rs)

    Args:
        p:
        q:
        r:
        s:
        n: the dimension(s) of the rank four tensor

    Returns:
        pq: int for the pq orbitals
        rs: int for the rs orbitals
    """
    pq = p * n + q
    rs = r * n + s
    return pq, rs


def lr_decomp(tbfo: FermionOperator) -> list[FermionOperator]:
    """Low Rank (LR) decomposition of the two-fermion part of the Hamiltonian.
    Implements the algorithm based on https://arxiv.org/pdf/1808.02625
    and https://pubs.acs.org/doi/10.1021/acs.jctc.7b00605.

    Procedure:
    1. reorder creation and annihilation operators of the two-body operator into two-body (V') and one-body parts.
    2. recast V' into a supermatrix indexed by orbitals (ps), (qr), involving electrons 1, 2 (?)
    3. decompose into rank-three auxiliary tensor L, via diagonalization or CD
    4. diagonalize/decompose each L^l
    5. gather fragments to get the double-factorized result

    Args:
        tbfo: two-body operator

    Returns:
        list of one-fermion fragments
    """
    reordered_tbt = reorder_operators_for_lr(tbfo)
    tbt = get_n_body_tensor(reordered_tbt)
    if len(set(tbt.shape)) != 1:
        raise ValueError("Expected the two-fermion tensor to be size N x N x N x N")
    tbt_supermatrix = make_supermatrix(tbt)
    e_ls, v_ls = sp.linalg.eigh(tbt_supermatrix)
    for i in range(v_ls.shape[1]):
        lambdas, v_us = sp.linalg.eigh(v_ls)


def frob_norm(tensor) -> float:
    """Returns the norm as defined by this formula:

    sqrt sum |tensor * tensor|

    Args:
        tensor: an N-rank tensor

    Returns:
        the value of the norm
    """
    return np.sqrt(np.sum(np.abs(tensor * tensor)))


def make_x_matrix(thetas: np.ndarray, n: int) -> np.ndarray:
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
    X = np.zeros((n, n))
    t = 0
    for x in range(n):
        for y in range(x + 1, n):
            X[y][x] = -thetas[t]
            X[x][y] = thetas[t]
            t += 1
    return X


def make_unitary(thetas, n: int) -> np.ndarray:
    return sp.linalg.expm(make_x_matrix(thetas, n))


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


def gfr_cost(lambdas, thetas, g_pqrs, n):
    t = n * (n + 1) / 2
    if lambdas.shape[0] != 1 and thetas.shape[0] != t - n:
        raise ValueError(
            "Expanded n * (n + 1) / 2 elements in lambdas and [n * (n + 1) / 2] -n elements in thetas"
        )
    w_pqrs = make_fr_tensor(lambdas, thetas, n)
    diff = g_pqrs - w_pqrs
    return np.sum(np.abs(diff * diff))


def gfro_decomp(
    tbt: np.ndarray,
    threshold=1e-6,
    max_iter: int = 10000,
    only_proceed_if_success: bool = False,
    seed: int = 0,
    debug: bool = False,
) -> list[GFROFragment]:
    """Greedy Full Rank Optimization (GFRO) as described by 'Hamiltonian Decomposition Techniques' by Smik Patel,
    and various Izmaylov group publications.

    Procedure:

    1. Introduce a G tensor. It is initialized to the original two-fermion tensor.
    2. Select an optimal fragment that minimizes the cost function, via non-linear gradient based optimization of lambda and theta.
    3. Update G tensor with chosen fragment, and recalculate L1 norm
    4. Repeat until L1 norm reaches the desired threshold

    Args:
        seed:
        only_proceed_if_success:
        threshold:
        max_iter:
        tbt: two-body operator in np.array form

    Returns:
        list of fragments
    """
    np.random.seed(seed)
    g_tensor = tbt.copy()
    frags: List[GFROFragment] = []
    iter = 0
    n = tbt.shape[0]
    while frob_norm(g_tensor) >= threshold and iter <= max_iter:
        factor = 10 / frob_norm(g_tensor)
        x_dim = n * (n + 1) // 2
        greedy_sol = try_find_greedy_fr_frag(n, threshold, g_tensor, factor, x_dim)
        if only_proceed_if_success:
            greedy_sol = retry_until_success(
                factor, g_tensor, greedy_sol, iter, n, threshold, x_dim
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
    return frags


def retry_until_success(factor, g_tensor, greedy_sol, iter, n, threshold, x_dim):
    tries = iter
    while not greedy_sol.success:
        warnings.warn(
            UserWarning(
                f"Failed to converge on iteration {iter}, trying again: {str(tries - iter)} try."
            )
        )
        greedy_sol = try_find_greedy_fr_frag(n, threshold, g_tensor, factor, x_dim)
        if tries > (100 + iter):
            raise ValueError("Couldn't find good greedy fragment")
        tries += 1
    return greedy_sol


def try_find_greedy_fr_frag(n, threshold, g_tensor, factor, x_dim) -> OptimizeResult:
    x0 = np.random.uniform(low=-1e-3, high=1e-3, size=(2 * x_dim) - n)
    greedy_sol: OptimizeResult = minimize(
        lambda x0: gfr_cost(x0[:x_dim], x0[x_dim:], factor * g_tensor, n),
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": 10000, "disp": False},
        tol=(threshold / n**4) ** 2,
    )
    return greedy_sol
