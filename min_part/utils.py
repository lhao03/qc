import json
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
from min_part.ham_decomp import gfro_fragment_occ
from min_part.operators import get_particle_number, get_total_spin, get_projected_spin
from min_part.plots import PlotNames
from min_part.tensor_utils import get_chem_tensors, obt2op
from min_part.typing import GFROFragment


@dataclass
class EnergyOccupation:
    energy: float
    spin_orbs: float


@dataclass
class LowerBoundConfig:
    xpoints: List[float]
    num_spin_orbs: int
    mol_name: str
    mol_of_interest: any
    stable_bond_length: float
    date: str


def choose_lowest_energy(
    eigenvalues,
    eigenvectors,
    num_spin_orbs,
    num_elecs,
    proj_spin,
    total_spin,
    debug=False,
) -> Tuple[float, float]:
    """Choose the minimum eigenvalue based on these constraints: number of electrons, projected spin and total spin.

    Assumes the eigenvectors is a matrix containing slater determinants.
    """
    possible_energies = []
    possible_energies_2 = []
    occs_spin = []
    for i in range(eigenvectors.shape[1]):
        e = eigenvalues[i]
        w = eigenvectors[:, i]
        if w.shape[0] == num_spin_orbs**2:
            n = get_particle_number(w, e=num_spin_orbs)
            s_2 = get_total_spin(w, num_spin_orbs // 2)
            s_z = get_projected_spin(w, num_spin_orbs // 2)
            occs_spin.append((e, n, s_2, s_z))
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
    if debug:
        print("new vector")
        for occ in occs_spin:
            print(f"{occ[0]}, elecs: {occ[1]}, spin: {occ[2]}{occ[3]}")
        print("===")
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
    with open(f"{file_name}", "rb") as pkl_file:
        frags = pickle.load(pkl_file)
    return frags


def diag_partitioned_fragments(
    h2_frags: List[FermionOperator] | List[GFROFragment],
    h1_v: np.ndarray,
    h1_w: np.ndarray,
    num_elecs: int,
    num_spin_orbs: int,
):
    n_2_energy = []
    n_2_spin_energy = []
    all_energies = []
    for i, frag in enumerate(h2_frags):
        if isinstance(frag, GFROFragment):
            occupations, eigenvalues = gfro_fragment_occ(
                fragment=frag, num_spin_orbs=num_spin_orbs
            )
            frag = frag.operators
        if len(frag.terms) > 0:
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
        h1_v, h1_w, num_spin_orbs, num_elecs, proj_spin=0, total_spin=0, debug=False
    )
    n2_final_energy = n2_h1_energy + sum(n_2_energy)
    n2_spin_final_energy = n2_spin_h1_energy + sum(n_2_spin_energy)
    print(f"====> fragment energy: {n_2_spin_energy}")
    return n2_final_energy, n2_spin_final_energy, all_final_energy


def get_saved_file_names(parent_dir) -> Tuple[List[str], List[str]]:
    gfro_child_dirs = os.listdir(os.path.join(parent_dir, "gfro"))
    lr_child_dirs = os.listdir(os.path.join(parent_dir, "lr"))
    sorted_gfro = sorted(
        [os.path.join(parent_dir, "gfro", c) for c in gfro_child_dirs],
        key=os.path.getctime,
    )
    sorted_lr = sorted(
        [os.path.join(parent_dir, "lr", c) for c in lr_child_dirs], key=os.path.getctime
    )
    return list(sorted_gfro), list(sorted_lr)


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


def save_energies(
    child_dir,
    config_settings,
    global_id,
    no_partitioning,
    lr_n_subspace_energies,
    gfro_n_subspace_energies,
    lr_n_s_subspace_energies,
    gfro_n_s_subspace_energies,
    lr_all_subspace_energies,
    gfro_all_subspace_energies,
):
    energies = {
        PlotNames.NO_PARTITIONING.value: no_partitioning,
        PlotNames.LR_N.value: lr_n_subspace_energies,
        PlotNames.GFRO_N.value: gfro_n_subspace_energies,
        PlotNames.LR_N_S.value: lr_n_s_subspace_energies,
        PlotNames.GFRO_N_S.value: gfro_n_s_subspace_energies,
        PlotNames.LR_F_SPACE.value: lr_all_subspace_energies,
        PlotNames.GFRO_F_SPACE.value: gfro_all_subspace_energies,
    }

    energies_json = json.dumps(energies)
    with open(
        os.path.join(child_dir, f"{config_settings.mol_name}_{str(global_id)}.json"),
        "w",
    ) as f:
        f.write(energies_json)


def load_energies(
    child_dir,
    config_settings,
    global_id,
):
    try:
        with open(
            os.path.join(
                child_dir, f"{config_settings.mol_name}_{str(global_id)}.json"
            ),
            "r",
        ) as file:
            data = json.load(file)
        return (
            data[PlotNames.NO_PARTITIONING.value],
            data[PlotNames.LR_N.value],
            data[PlotNames.GFRO_N.value],
            data[PlotNames.LR_N_S.value],
            data[PlotNames.GFRO_N_S.value],
            data[PlotNames.LR_F_SPACE.value],
            data[PlotNames.GFRO_F_SPACE.value],
        )
    except FileNotFoundError:
        warnings.warn("JSON not found, continuing with empty arrays.")
    return [], [], [], [], [], [], []


def closetoin(i, arr):
    for a in arr:
        if isclose(i, a):
            return True
    return False
