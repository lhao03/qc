import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy as sp
from openfermion import FermionOperator, qubit_operator_sparse, jordan_wigner

from d_types.config_types import MConfig
from d_types.fragment_types import (
    FermionicFragment,
    Subspace,
    PartitionStrategy,
    GFROFragment,
    LRFragment,
)
from min_part.gfro_decomp import gfro_decomp, gfro_fragment_occ
from min_part.lr_decomp import lr_decomp
from min_part.tensor import obt2op, tbt2op


def zero_total_spin(t: Tuple[int]):
    even = 0
    odd = 0
    for x in t:
        if x % 2 == 0:
            even += 1
        else:
            odd += 1
    return even == odd


@dataclass
class FragmentedHamiltonian:
    m_config: MConfig
    constant: float
    one_body: FermionicFragment | np.ndarray
    two_body: List[FermionicFragment] | np.ndarray
    partitioned: bool
    fluid: bool
    subspace: Subspace

    def __eq__(self, other):
        if isinstance(other, FragmentedHamiltonian):
            ob_eq = (
                np.allclose(self.one_body, other.one_body)
                if isinstance(self.one_body, np.ndarray)
                else self.one_body == other.one_body
            )
            tb_eq = (
                np.allclose(self.two_body, other.two_body)
                if isinstance(self.two_body, np.ndarray)
                else self.two_body == other.two_body
            )
            config_eq = self.m_config == self.m_config
            constant_eq = self.constant == other.constant
            partitioned_eq = self.partitioned == other.partitioned
            fluid_eq = self.fluid == other.fluid
            subspace_eq = self.subspace == other.subspace
            return (
                config_eq
                and constant_eq
                and ob_eq
                and tb_eq
                and partitioned_eq
                and fluid_eq
                and subspace_eq
            )
        else:
            return False

    def _diagonalize_operator(self, fo: FermionOperator):
        eigenvalues, eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(fo)).toarray()
        )
        subspace_w = filter(
            lambda i_w: (
                self.subspace.n(i_w[1]) == self.subspace.expected_e
                and self.subspace.sz(i_w[1]) == self.subspace.expected_sz
                and self.subspace.s2(i_w[1]) == self.subspace.expected_s2
            ),
            enumerate(eigenvectors.T),
        )
        subspace_e = [eigenvalues[i_w[0]] for i_w in subspace_w]
        return min(subspace_e)

    def partition(self, strategy: PartitionStrategy):
        match strategy:
            case PartitionStrategy.GFRO:
                self.two_body = gfro_decomp(self.two_body)
            case PartitionStrategy.LR:
                self.two_body = lr_decomp(self.two_body)
        self.partitioned = True
        return self.two_body

    def get_expectation_value(self):
        if self.partitioned and self.fluid:
            pass
        elif self.partitioned and not self.fluid:
            const_obt = self._diagonalize_operator(
                self.constant + obt2op(self.one_body)
            )
            tbt_e = 0
            for frag in self.two_body:
                if isinstance(frag, LRFragment):
                    pass
                elif isinstance(frag, GFROFragment):
                    occs, energies = gfro_fragment_occ(
                        fragment=frag,
                        num_spin_orbs=self.m_config.num_spin_orbs,
                        occ=self.subspace.expected_e,
                    )
                    subspace_energy = filter(
                        lambda occ_ener: zero_total_spin(occ_ener[0]),
                        zip(occs, energies),
                    )
                    tbt_e += min(subspace_energy, default=0)
                else:
                    raise UserWarning("Should not end up here.")
            return const_obt + tbt_e
        elif not self.partitioned and not self.fluid:
            if not (
                isinstance(self.one_body, np.ndarray)
                and isinstance(self.two_body, np.ndarray)
            ):
                raise UserWarning(
                    "Expected one-electron and two-electron parts to be tensors."
                )
            return self._diagonalize_operator(
                self.constant + obt2op(self.one_body) + tbt2op(self.two_body)
            )
        else:
            raise UserWarning("Shouldn't end up here.")

    def save(self):
        file_name = os.path.join(
            self.m_config.folder, self.m_config.mol_name, self.m_config.date
        )
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        if self.partitioned and self.fluid:
            file_name = os.path.join(
                file_name,
                (
                    "fluid_gfro"
                    if isinstance(self.two_body[0], GFROFragment)
                    else "fluid_lr"
                ),
            )
        elif self.partitioned and not self.fluid:
            file_name = os.path.join(
                file_name,
                ("gfro" if isinstance(self.two_body[0], GFROFragment) else "lr"),
            )
        elif not self.partitioned and not self.fluid:
            file_name = os.path.join(
                file_name,
                "unpartitioned",
            )
        else:
            raise UserWarning("Shouldn't end up here.")
        output = open(f"{file_name}.pkl", "wb")
        pickle.dump(self, output)
        output.close()
        return file_name

    @classmethod
    def load(cls, file_name):
        if (
            "fluid_gfro" in file_name
            or "fluid_lr" in file_name
            or "gfro" in file_name
            or "lr" in file_name
            or "unpartitioned" in file_name
        ):
            with open(f"{file_name}.pkl", "rb") as pkl_file:
                frags = pickle.load(pkl_file)
            return frags
        else:
            raise UserWarning(
                "Expected filename to contain 'fluid', 'gfro', 'lr', or 'unpartitioned'"
            )
