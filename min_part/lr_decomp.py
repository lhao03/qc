from typing import Any, List

import numpy as np
import scipy as sp
from openfermion import FermionOperator
from opt_einsum import contract

from d_types.fragment_types import LRFragment
from min_part.f_3_ops import extract_thetas
from min_part.julia_ops import lr_decomp_params
from min_part.reorder import reorder_operators_for_lr
from min_part.tensor import get_n_body_tensor
from min_part.tensor_utils import tbt2op


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
        n: the dimension of the rank four tensor

    Returns:
        pq: int for the pq orbitals
        rs: int for the rs orbitals
    """
    pq = p * n + q
    rs = r * n + s
    return pq, rs


def lr_decomp(tbt: np.ndarray) -> list[LRFragment]:
    """Low Rank (LR) decomposition of the two-fermion part of the Hamiltonian.
    Implements the algorithm based on https://arxiv.org/pdf/1808.02625

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
    lr_frags = lr_decomp_params(tbt)
    lr_frag_details = []
    for i, lr_frag in enumerate(lr_frags):
        lambdas = lr_frag[0]
        u = lr_frag[1]
        tensor = contract("ij,pi,qi,rj,sj -> pqrs", lambdas, u, u, u, u)
        operators = tbt2op(tensor)
        if operators.induced_norm(2) > 1e-6:
            lr_frag_details.append(
                LRFragment(
                    lambdas=lambdas,
                    operators=operators,
                    thetas=extract_thetas(u),
                )
            )
    return lr_frag_details


def lr_fragment_occ(
    fragment, num_spin_orbs: int, occupied_spin_orbs: List[int]
) -> List[float]:
    pass
