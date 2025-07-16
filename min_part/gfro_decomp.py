import warnings
from typing import List, Optional

import numpy as np
from opt_einsum import contract
from scipy.optimize import OptimizeResult, minimize

from d_types.fragment_types import GFROFragment
from min_part.tensor import tbt2op, make_lambda_matrix
from d_types.unitary_type import make_unitary, ReaDeconUnitary


def frob_norm(tensor) -> float:
    """Returns the norm as defined by this formula:

    sqrt sum |tensor * tensor|

    Args:
        tensor: an N-rank tensor

    Returns:
        the value of the norm
    """
    return np.sqrt(np.sum(np.abs(tensor * tensor)))


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
    return contract("lm,lp,lq,mr,ms->pqrs", lm, U, U, U, U)  # TODO: check??


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
    debug: bool = False,
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
                lambdas=lambdas_sol,
                unitary=ReaDeconUnitary(thetas=thetas_sol, dim=n),
                operators=tbt2op(fr_frag_tensor),
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
