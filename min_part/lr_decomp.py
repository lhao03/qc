from typing import Any, List

import numpy as np
from opt_einsum import contract

from d_types.fragment_types import LRFragment, Nums
from min_part.julia_ops import lr_decomp_params
from min_part.tensor import tbt2op, extract_thetas, make_unitary_im


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

    Args:
        tbfo: two-body operator

    Returns:
        list of one-fermion fragments
    """
    lr_frags = lr_decomp_params(tbt)
    lr_frag_details = []
    for i, lr_frag in enumerate(lr_frags):
        outer_coeff, coeffs = lr_frag[0]
        u = lr_frag[1]
        thetas, diags = extract_thetas(u)
        tensor = lr_frag[2]
        operators = tbt2op(tensor)
        if operators.induced_norm(2) > 1e-6:
            lr_frag_details.append(
                LRFragment(
                    outer_coeff=outer_coeff,
                    coeffs=coeffs,
                    operators=operators,
                    thetas=thetas,
                    diag_coeffs=diags,
                )
            )
    return lr_frag_details


def get_lr_fragment_tensor(lr_details: LRFragment):
    return get_lr_fragment_tensor_from_parts(
        outer_coeff=lr_details.outer_coeff,
        coeffs=lr_details.coeffs,
        thetas=lr_details.thetas,
        diags=lr_details.diag_coeffs,
    )


def get_lr_fragment_tensor_from_parts(
    outer_coeff: float, coeffs: Nums, thetas: Nums, diags: Nums
):
    coeffs = np.array(coeffs)
    u = make_unitary_im(thetas=thetas, diags=diags, n=diags.size)
    return contract("ij,pi,qi,rj,sj->pqrs", outer_coeff * coeffs @ coeffs.T, u, u, u, u)


def lr_fragment_occ(
    fragment, num_spin_orbs: int, occupied_spin_orbs: List[int]
) -> List[float]:
    pass
