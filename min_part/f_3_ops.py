from typing import Tuple, Optional

import numpy as np
from openfermion import FermionOperator
from opt_einsum import contract

from d_types.fragment_types import (
    FluidFermionicFragment,
    FermionicFragment,
    LRFragment,
    GFROFragment,
    Nums,
)
from min_part.gfro_decomp import (
    make_lambda_matrix,
    make_unitary,
)
from min_part.julia_ops import solve_quad
from min_part.tensor_utils import tbt2op


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
        curr_i += (j + 1)
    assert len(ob) == n
    return ob

def remove_one_body_parts(lambdas: Nums) -> Nums:
    n = solve_quad(1, 1, -2 * lambdas.size)
    curr_i = 0
    for j in reversed(range(n)):
        lambdas[curr_i] = 0
        curr_i += (j + 1)
    return lambdas

def oneb2op(diags: Nums, thetas) -> FermionOperator:
    """Makes the `FermionOperator` Object from one-body parts.

    Args:
        diags: coefficients for each spin orbital. Assuming they are in increasing order of spin orbital.

    Returns:
         `FermionOperator` containing only one body parts
    """
    diags = np.array(diags)
    n = diags.size
    l_mat = np.diagflat(diags)
    unitary = make_unitary(thetas, n)
    tbt_as_obt = contract("lm,lp,lq,mr,ms->pqrs", l_mat, unitary, unitary, unitary, unitary)
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

def fragment2fluid(frag: FermionicFragment, performant: bool = False) -> FluidFermionicFragment:
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
            if not performant:
                assert operators == oneb2op(fluid_frags) + twob2op(static_frags)
            return FluidFermionicFragment(
                static_frags=static_frags,
                fluid_frags=fluid_frags,
                thetas=thetas,
                operators=None
            )
    pass


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

def rediag_onebody(undiagonalized_onebody: FluidFermionicFragment) -> LRFragment:
    pass
