from typing import Any, Optional, Tuple

import numpy as np
from opt_einsum import contract

from d_types.config_types import Basis
from d_types.fragment_types import LRFragment, Nums
from min_part.f3_opers import lambdas_from_fluid_parts
from min_part.julia_ops import lr_decomp_params
from min_part.operators import generate_occupied_spin_orb_permutations
from min_part.tensor import tbt2op, make_lambda_matrix
from d_types.unitary_type import Unitary, WholeUnitary


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
        operators = tbt2op(lr_frag[2])
        if operators.induced_norm(2) > 1e-6:
            outer_coeff, coeffs = lr_frag[0]
            u = lr_frag[1]
            lr_frag_details.append(
                LRFragment(
                    outer_coeff=outer_coeff,
                    coeffs=coeffs,
                    operators=operators,
                    unitary=WholeUnitary(mat=u, dim=u.shape[0], basis=Basis.SPIN),
                    basis=Basis.SPIN,
                )
            )
    return lr_frag_details


def get_lr_fragment_tensor(lr_details: LRFragment):
    return get_lr_fragment_tensor_from_parts(
        outer_coeff=lr_details.outer_coeff,
        coeffs=lr_details.coeffs,
        unitary=lr_details.unitary,
    )


def get_lr_fragment_tensor_from_parts(
    outer_coeff: float, coeffs: Nums, unitary: Unitary
):
    coeffs = np.array(coeffs)
    u = unitary.make_unitary_matrix()
    return contract("ij,pi,qi,rj,sj->pqrs", outer_coeff * coeffs @ coeffs.T, u, u, u, u)


def get_lr_fragment_tensor_from_lambda(lambdas: Nums, unitary: Unitary, n: int):
    u = unitary.make_unitary_matrix()
    return contract("ij,pi,qi,rj,sj->pqrs", make_lambda_matrix(lambdas, n), u, u, u, u)


def lr_fragment_occ(
    fragment: LRFragment, num_spin_orbs: int, occ: Optional[int] = None
) -> Tuple[list[Tuple[int]], np.ndarray[Any, np.dtype[Any]]]:
    occupation_combinations = generate_occupied_spin_orb_permutations(
        num_spin_orbs, occ
    )
    occ_energies = []
    for occ_comb in occupation_combinations:
        occ_energy = 0
        for l in occ_comb:
            for m in occ_comb:
                occ_energy += float(fragment.coeffs[l]) * float(fragment.coeffs[m])
        occ_energies.append(fragment.outer_coeff * occ_energy)
    return occupation_combinations, np.array(occ_energies)


def lr_fragment_occ_from_lambdas(
    lambdas: np.ndarray, num_spin_orbs: int, occ: Optional[int] = None
) -> tuple[list[tuple[int]], list[int | Any]]:
    occupation_combinations = generate_occupied_spin_orb_permutations(
        num_spin_orbs, occ
    )
    l_mat = make_lambda_matrix(lambdas, num_spin_orbs)
    occ_energies = []
    for occ_comb in occupation_combinations:
        occ_energy = 0
        for l in occ_comb:
            for m in occ_comb:
                occ_energy += l_mat[l, m]
        occ_energies.append(occ_energy)
    return occupation_combinations, occ_energies


def get_expectation_vals_lr_frags(
    self: LRFragment, num_spin_orbs: int, expected_e: int
):
    if self.fluid_parts is None:
        return lr_fragment_occ(self, num_spin_orbs, expected_e)
    else:
        lambdas = lambdas_from_fluid_parts(self.fluid_parts)
        return lr_fragment_occ_from_lambdas(lambdas, num_spin_orbs, expected_e)
