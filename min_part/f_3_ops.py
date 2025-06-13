from typing import Tuple, Optional

import numpy as np
from openfermion import FermionOperator
from opt_einsum import contract
import scipy as sp
from d_types.fragment_types import (
    FluidFermionicFragment,
    FermionicFragment,
    LRFragment,
    GFROFragment,
    Nums,
    FluidCoeff,
)
from min_part.gfro_decomp import (
    make_lambda_matrix,
    make_unitary,
)
from min_part.julia_ops import solve_quad, eigen_jl
from min_part.tensor_utils import tbt2op, obt2op


def get_one_body_parts(lambdas: Nums) -> Nums:
    """Returns the one-body part from a lambda matrix formed after LR or GFRO decomposition.
    Args:
        lambdas: list of amplitudes that form a lambda matrix

    Returns:
        the one body coefficients in ascending order, 1, 2, 3...n
    """
    n = solve_quad(1, 1, -2 * lambdas.size)
    curr_i = 0
    ob = []
    for j in reversed(range(n)):
        ob.append(lambdas[curr_i])
        curr_i += j + 1
    assert len(ob) == n
    return ob


def remove_one_body_parts(lambdas: Nums) -> Nums:
    n = solve_quad(1, 1, -2 * lambdas.size)
    curr_i = 0
    for j in reversed(range(n)):
        lambdas[curr_i] = 0
        curr_i += j + 1
    return lambdas


def oneb2op(fluid_coeffs: FluidCoeff) -> FermionOperator:
    """Makes the `FermionOperator` Object from one-body parts.

    Args:
        diags: coefficients for each spin orbital. Assuming they are in increasing order of spin orbital.

    Returns:
         `FermionOperator` containing only one body parts
    """
    diags = np.array(fluid_coeffs.coeff)
    n = diags.size
    l_mat = np.diagflat(diags)
    unitary = make_unitary(fluid_coeffs.thetas, n)
    tbt_as_obt = contract(
        "lm,lp,lq,mr,ms->pqrs", l_mat, unitary, unitary, unitary, unitary
    )
    return tbt2op(tbt_as_obt)


def twob2op(lambdas: Nums, thetas) -> FermionOperator:
    """Makes the `FermionOperator` Object from two-body parts.

    Args:
        diags: coefficients for each spin orbital. Assuming they are in increasing order of spin orbital.

    Returns:
         `FermionOperator` containing only one body parts
    """
    lambdas = np.array(lambdas)
    n = solve_quad(1, 1, -2 * lambdas.size)
    l_mat = make_lambda_matrix(lambdas, n)
    unitary = make_unitary(thetas, n)
    tbt = contract("lm,lp,lq,mr,ms->pqrs", l_mat, unitary, unitary, unitary, unitary)
    return tbt2op(tbt)


def fragment2fluid(
    frag: FermionicFragment, performant: bool = False
) -> FluidFermionicFragment:
    """Converts any FermionicFragment into a FluidFermionicFragment, by separating out the one-body part from the two-body part,
    if possible.

    Args:
        performant: whether or not to perform extra checks. Setting this to True performs operator and equality checks.
        frag: A fragment generated from GFRO or LR procedure.

    Returns:
        A fragment type ready for optimization as a fluid fermionic fragment.
    """
    match frag:
        case GFROFragment(thetas, operators, lambdas):
            fluid_frags = get_one_body_parts(lambdas)
            static_frags = remove_one_body_parts(lambdas)
            fluid_coeff = FluidCoeff(coeff=fluid_frags, thetas=thetas)
            if not performant:
                assert operators == oneb2op(fluid_coeff) + twob2op(static_frags, thetas)
            return FluidFermionicFragment(
                static_frags=static_frags,
                fluid_frags=[FluidCoeff(coeff=fluid_frags, thetas=thetas)],
                thetas=thetas,
                operators=operators,
            )


def move_onebody_coeff(
    from_frag: FluidFermionicFragment,
    to_frag: FluidFermionicFragment,
    coeff: float,
    mutate: bool = True,
) -> Optional[Tuple[FluidFermionicFragment, FluidFermionicFragment]]:
    pass


def make_super_unitary_matrix():
    pass


def make_augmented_hpq_matrix():
    pass


def extract_thetas(U) -> Nums:
    """Extracts theta values from a unitary matrix paramertized by amplitudes.
    Args:
        U: the unitary

    Returns:
        theta values
    """
    X, _ = sp.linalg.logm(U, disp=False)
    thetas = []
    u = U.shape[0]
    for i in range(u):
        for j in range(i + 1, u):
            thetas.append(X[i, j])
    return thetas


def obt2fluid(obt: np.ndarray) -> FluidFermionicFragment:
    """
    Converts a one-body tensor to a `FluidFermionicFragment` type via diagonalization of the tensor.
    Args:
        obt:

    Returns:
        `FluidFermionicFragment` object of the one-body tensor
    """
    U, V = eigen_jl(obt)
    assert V.size == obt.shape[0]
    assert U.shape == obt.shape
    thetas = extract_thetas(U)
    return FluidFermionicFragment(
        thetas=thetas, fluid_frags=[], static_frags=V, operators=obt2op(obt)
    )


def rediag_onebody(undiagonalized_onebody: FluidFermionicFragment) -> LRFragment:
    pass


def obf3to_op(lambdas: np.ndarray, thetas) -> FermionOperator:
    n = lambdas.size
    l_mat = np.diagflat(lambdas)
    unitary = make_unitary(thetas, n)
    obt = np.einsum("ab,ap,bq->pq", l_mat, unitary, unitary)
    return obt2op(obt)
