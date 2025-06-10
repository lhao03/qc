from typing import Tuple, Optional, List

import numpy as np

from d_types.fragment_types import FluidFermionicFragment, FermionicFragment, LRFragment, GFROFragment
from min_part.julia_ops import solve_quad


def get_one_body_parts(thetas: np.ndarray) -> List[float]:
    """Returns the one-body part from a lambda matrix formed after LR or GFRO decomposition.
    Args:
        thetas: list of amplitudes that form a lambda matrix

    Returns:
        the one body coefficients in ascending order, 1, 2, 3...n
    """
    n = solve_quad(1, 1, -2 * thetas.size)
    curr_i = 0
    ob = []
    for j in reversed(range(n)):
        ob.append(thetas[curr_i])
        curr_i += (j + 1)
    assert len(ob) == n
    return ob

def fragment2fluid(frag: FermionicFragment) -> FluidFermionicFragment:
    """Converts any FermionicFragment into a FluidFermionicFragment, by separating out the one-body part from the two-body part,
    if possible.

    Args:
        frag: A fragment generated from GFRO or LR procedure.

    Returns:
        A fragment type ready for optimization as a fluid fermionic fragment.
    """
    match frag:
        case GFROFragment(thetas, operators, lambdas):
            return FluidFermionicFragment()
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
