import os
import pickle
import warnings
from dataclasses import dataclass
from enum import Enum

import scipy as sp
from math import isclose
from typing import Optional, List, Tuple

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
) -> Tuple[float, float]:
    """Choose the minimum eigenvalue based on these constraints: number of electrons, projected spin and total spin.

    Assumes the eigenvectors is a matrix containing slater determinants. Currently, can only
    """
    possible_energies = []
    possible_energies_2 = []
    for i in range(eigenvectors.shape[1]):
        e = eigenvalues[i]
        w = eigenvectors[:, i]
        n = get_particle_number(w, e=num_spin_orbs)
        s_2 = get_total_spin(w, num_spin_orbs // 2)
        s_z = get_projected_spin(w, num_spin_orbs // 2)
        if (
            isclose(n, num_elecs, abs_tol=1e-6)
            # and isclose(s_2, total_spin, abs_tol=1e-6)
            # and isclose(s_z, proj_spin, abs_tol=1e-6)
        ):
            possible_energies.append(e)
            if isclose(s_2, total_spin, abs_tol=1e-6) and isclose(
                s_z, proj_spin, abs_tol=1e-6
            ):
                possible_energies_2.append(e)
    if len(possible_energies) == 0:
        warnings.warn(
            UserWarning("Returning 0 energy value, no values to filter from.")
        )
    return (
        min(possible_energies, default=0),
        min(possible_energies_2, default=0),
    )


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
    n_2_spin_energy = []
    all_energies = []
    for i, frag in enumerate(h2_frags):
        eigenvalues, eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(frag)).toarray()
        )
        n_2, n_2_spin = choose_lowest_energy(
            eigenvalues,
            eigenvectors,
            num_spin_orbs,
            num_elecs,
            proj_spin=0,
            total_spin=0,
        )
        n_2_energy.append(n_2)
        n_2_spin_energy.append(n_2_spin)
        all_energies.append(min(eigenvalues))

    # === all of fock space ===
    all_final_energy = min(h1_v) + sum(all_energies)

    # === projected onto gs ===
    n2_h1_energy, n2_spin_h1_energy = choose_lowest_energy(
        h1_v, h1_w, num_spin_orbs, num_elecs, proj_spin=0, total_spin=0
    )
    n2_final_energy = n2_h1_energy + sum(n_2_energy)
    n2_spin_final_energy = n2_spin_h1_energy + sum(n_2_spin_energy)

    return n2_final_energy, n2_spin_final_energy, all_final_energy


def get_saved_file_names(
    parent_dir, gfro_file_name, lr_file_name
) -> Tuple[List[str], List[str]]:
    gfro_files = []
    lr_files = []
    child_dirs = os.listdir(parent_dir)
    sorted_parent_dirs = sorted(
        [os.path.join(parent_dir, c) for c in child_dirs], key=os.path.getctime
    )
    for dir in sorted_parent_dirs:
        gfro_files.append(os.path.join(dir, gfro_file_name))
        lr_files.append(os.path.join(dir, lr_file_name))
    return gfro_files, lr_files


class PartitionStrategy(Enum):
    GFRO = "GFRO"
    LR = "LR"


@dataclass
class PartitioningStats:
    bond_length: float
    num_frags: int
    e_diff: float
    partition_strategy: PartitionStrategy


def partitioning_stats(
    no_partitioning_energy: float,
    partitioned_energy: float,
    frags: list,
    bond_length: float,
    partition_strategy: PartitionStrategy,
):
    return PartitioningStats(
        bond_length=bond_length,
        num_frags=len(frags),
        e_diff=no_partitioning_energy - partitioned_energy,
        partition_strategy=partition_strategy,
    )


def range_float(start, end, step):
    return [x / 10.0 for x in range(int(start * 10), int(end * 10), int(step * 10))]
