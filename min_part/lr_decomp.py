from typing import Any, List

import numpy as np
import scipy as sp
from openfermion import FermionOperator

from min_part.reorder import reorder_operators_for_lr
from min_part.tensor import get_n_body_tensor


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


def lr_fragment_occ(
    fragment, num_spin_orbs: int, occupied_spin_orbs: List[int]
) -> List[float]:
    pass
