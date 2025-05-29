import pickle
from dataclasses import dataclass

import scipy as sp
from math import isclose
from typing import Optional, List

import numpy as np
from openfermion import (
    FermionOperator,
    jordan_wigner,
    qubit_operator_sparse,
)

from min_part.ffrag_utils import LR_frags_generator
from min_part.operators import get_particle_number, get_total_spin, get_projected_spin
from min_part.tensor_utils import get_chem_tensors, obt2op


@dataclass
class EnergyOccupation:
    energy: float
    spin_orbs: float


def choose_lowest_energy(
    eigenvalues, eigenvectors, num_spin_orbs, num_elecs, proj_spin, total_spin
) -> float:
    """Choose the minimum eigenvalue based on these constraints: number of electrons, projected spin and total spin.

    Assumes the eigenvectors is a matrix containing slater determinants. Currently, can only
    """
    possible_energies = []
    for i in range(eigenvectors.shape[1]):
        e = eigenvalues[i]
        w = eigenvectors[:, i]
        n = get_particle_number(w, e=num_spin_orbs)
        s_2 = get_total_spin(w, num_spin_orbs // 2)
        s_z = get_projected_spin(w, num_spin_orbs // 2)
        if (
            isclose(n, num_elecs, abs_tol=1e-6)
            and isclose(s_2, total_spin, abs_tol=1e-6)
            and isclose(s_z, proj_spin, abs_tol=1e-6)
        ):
            possible_energies.append(e)
    if len(possible_energies) == 0:
        raise UserWarning("Returning 0 energy value, no values to filter from.")
    return min(possible_energies, default=0)


def dc_to_dict(dcs, labels: list[str]):
    attrs = list(dcs[0].__dict__.keys())
    tb_dict = {}
    for i, label in enumerate(labels):
        tb_dict[label] = []
        for dc in dcs:
            tb_dict[label].append(getattr(dc, attrs[i]))
    return tb_dict


def do_lr_fo(
    H_FO: FermionOperator,
    projector_func: Optional = None,
    project: bool = False,
):
    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)

    # Obtaining LR fragments as list of FermionOperators and (coeffs, angles) defining the fragments.
    lowrank_fragments, lowrank_params = LR_frags_generator(
        tbt, tol=1e-5, ret_params=True
    )

    # Filtering out small fragments
    LR_fragments = []
    LR_params = []
    for i in range(len(lowrank_params)):
        frag = lowrank_fragments[i]
        if frag.induced_norm(2) > 1e-6:
            LR_fragments.append(frag)
            LR_params.append(lowrank_params[i])
    all_frag_ops = [const * FermionOperator.identity(), obt_op]
    all_frag_ops += LR_fragments

    if project:
        return (
            projector_func(const * FermionOperator.identity(), excitation_level=None),
            projector_func(obt_op, excitation_level=None),
            [projector_func(lr_f, excitation_level=None) for lr_f in LR_fragments],
        )
    return const * FermionOperator.identity(), obt_op, LR_fragments


def save_frags(frags, file_name):
    output = open(f"{file_name}.pkl", "wb")
    pickle.dump(frags, output)
    output.close()


def open_frags(file_name):
    pkl_file = open(f"{file_name}.pkl", "rb")
    return pickle.load(pkl_file)


def diag_partitioned_fragments(
    h2_frags: List[FermionOperator],
    h1_v: np.ndarray,
    h1_w: np.ndarray,
    num_elecs: int,
    num_spin_orbs: int,
):
    n_2_energy = []
    all_energies = []
    for frag in h2_frags:
        eigenvalues, eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(frag)).toarray()
        )
        n_2_energy.append(
            choose_lowest_energy(
                eigenvalues,
                eigenvectors,
                num_spin_orbs,
                num_elecs,
                proj_spin=0,
                total_spin=0,
            )
        )
        all_energies.append(min(eigenvalues))

    # === all of fock space ===
    all_final_energy = min(h1_v) + sum(all_energies)

    # === projected onto gs ===
    n2_h1_energy = choose_lowest_energy(
        h1_v, h1_w, num_spin_orbs, num_elecs, proj_spin=0, total_spin=0
    )
    n2_final_energy = n2_h1_energy + sum(n_2_energy)

    return n2_final_energy, all_final_energy
