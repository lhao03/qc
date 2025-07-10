import os
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
from openfermion import (
    FermionOperator,
    qubit_operator_sparse,
    jordan_wigner,
    number_operator,
    s_squared_operator,
    sz_operator,
    expectation,
)

from d_types.config_types import MConfig
from d_types.cvx_exp import make_fluid_variables
from d_types.fragment_types import (
    FermionicFragment,
    Subspace,
    PartitionStrategy,
    GFROFragment,
    OneBodyFragment,
)
from min_part.f3_opers import obt2fluid
from min_part.gfro_decomp import gfro_decomp
from min_part.lr_decomp import lr_decomp
from min_part.operators import subspace_projection_operator
from min_part.tensor import obt2op, tbt2op
from min_part.utils import open_frags, save_frags


class OptType(Enum):
    OFAO = "OFAO"
    OFAT = "OFAT"
    GREEDY = "GREEDY"
    CONVEX = "CONVEX"


def zero_s_z(t: Tuple[int]):
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
    one_body: OneBodyFragment | np.ndarray
    two_body: List[FermionicFragment] | np.ndarray
    partitioned: bool
    fluid: bool
    subspace: Subspace
    number_operator: any = None
    s2_operator: any = None
    sz_operator: any = None
    ci_projection: Optional[int] = None
    fluid_variables: any = None

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

    def _return_proper_fluid_vars(self, frag):
        n = self.m_config.num_spin_orbs
        for i, f in enumerate(self.two_body):
            if f == frag:
                return self.fluid_variables[i * n, (i * n) + n]

    def optimize_fragments(
        self,
        optimization_type: OptType,
        desired_occs: Optional = None,
        iters: int = 1000,
        debug: bool = False,
    ):
        self.fluid_variables = make_fluid_variables(
            n=self.one_body.lambdas.shape[0], self=self
        )
        match optimization_type:
            case OptType.OFAT:
                return ofat_fluid_optimize(self, iters=iters, debug=debug)
            case OptType.OFAO:
                return afao_fluid_optimize(self, iters=iters)
            case OptType.GREEDY:
                return greedy_coeff_optimize(self, iters=10000, threshold=1e-9)
            case OptType.CONVEX:
                return convex_optimization(self, desired_occs)

    def _add_up_orb_occs(self, frag: FermionicFragment):
        occs, energies = frag.get_expectation_value(
            num_spin_orbs=self.m_config.num_spin_orbs,
            expected_e=self.subspace.expected_e,
        )
        subspace_energy = filter(
            lambda occ_ener: zero_s_z(occ_ener[0]),
            zip(occs, energies),
        )
        return min([o_e[1] for o_e in subspace_energy], default=0)

    def _trace(self, fo: FermionOperator, complete: bool = False):
        eigenvalues, eigenvectors = (
            np.linalg.eigh(qubit_operator_sparse(jordan_wigner(fo)).toarray())
            if complete
            else np.linalg.eigh(
                subspace_projection_operator(
                    fo, self.m_config.num_spin_orbs, self.subspace.expected_e
                ).toarray()
            )
        )
        return sum(eigenvalues)

    def _diagonalize_operator_manual_ss(self, fo: FermionOperator):
        eigs, vecs = np.linalg.eigh(qubit_operator_sparse(jordan_wigner(fo)).toarray())
        if (
            not isinstance(self.number_operator, np.ndarray)
            or not isinstance(self.s2_operator, np.ndarray)
            or not isinstance(self.sz_operator, np.ndarray)
        ):
            self.number_operator = qubit_operator_sparse(
                jordan_wigner(number_operator(n_modes=self.m_config.num_spin_orbs))
            )
            self.s2_operator = qubit_operator_sparse(
                jordan_wigner(
                    s_squared_operator(
                        n_spatial_orbitals=self.m_config.num_spin_orbs // 2
                    )
                )
            )
            self.sz_operator = qubit_operator_sparse(
                jordan_wigner(
                    sz_operator(n_spatial_orbitals=self.m_config.num_spin_orbs // 2)
                )
            )
        energies = []
        for i in range(vecs.shape[0]):
            vec = vecs[:, i]
            num_elecs = round(expectation(operator=self.number_operator, state=vec))
            s2 = expectation(operator=self.s2_operator, state=vec)
            sz = expectation(operator=self.sz_operator, state=vec)
            if (
                (num_elecs == self.subspace.expected_e)
                and (np.isclose(self.subspace.expected_s2, s2))
                and (np.isclose(self.subspace.expected_sz, sz))
            ):
                energies.append(eigs[i])
        return min(energies)

    def _diagonalize_operator_with_ss_proj(self, fo: FermionOperator):
        eigenvalues, eigenvectors = np.linalg.eigh(
            self.subspace.projector(fo).toarray()
        )
        return min(eigenvalues, default=0)

    def _filter_frag_energy(
        self, frag: FermionicFragment, desired_occs: Optional[Tuple] = None
    ):
        occs, energies = frag.get_expectation_value(
            num_spin_orbs=self.m_config.num_spin_orbs,
            expected_e=self.subspace.expected_e,
        )
        if desired_occs is not None:
            desired_energies = []
            for occ, energy in zip(occs, energies):
                if occ in desired_occs:
                    desired_energies.append(energy)
            return min(desired_energies)
        else:
            subspace_energy = filter(
                lambda occ_ener: zero_s_z(occ_ener[0]),
                zip(occs, energies),
            )
            return min([o_e[1] for o_e in subspace_energy], default=0)

    def get_operators(self):
        ob_op = (
            obt2op(self.one_body)
            if isinstance(self.one_body, np.ndarray)
            else self.one_body.to_op()
        )
        tb_op = (
            tbt2op(self.two_body)
            if isinstance(self.two_body, np.ndarray)
            else sum([f.to_op() for f in self.two_body])
        )
        return self.constant + ob_op + tb_op

    def partition(
        self, strategy: PartitionStrategy, bond_length: float, load_prev: bool = True
    ):
        frag_path = os.path.join(
            self.m_config.frag_folder,
            ("gfro" if strategy is PartitionStrategy.GFRO else "lr"),
            str(bond_length),
        )
        if load_prev and os.path.exists(f"{frag_path}.pkl"):
            self.two_body = open_frags(frag_path)
        else:
            match strategy:
                case PartitionStrategy.GFRO:
                    self.two_body = gfro_decomp(self.two_body, debug=True)
                case PartitionStrategy.LR:
                    self.two_body = lr_decomp(self.two_body)
            save_frags(self.two_body, file_name=frag_path)
        self.partitioned = True
        self.one_body = obt2fluid(self.one_body)
        return self.two_body

    def get_expectation_value(
        self,
        use_frag_energies: bool = False,
        diag_complete_space: bool = False,
        desired_occs: Optional = None,
    ):
        if not self.subspace.projector:
            self.subspace.projector = partial(
                subspace_projection_operator,
                n_spin_orbs=self.m_config.num_spin_orbs,
                num_elecs=self.subspace.expected_e,
                ci_projection=self.ci_projection,
            )
        if self.partitioned or self.fluid:
            const_obt = self._diagonalize_operator_with_ss_proj(
                self.constant + self.one_body.to_op()
            )
            tbt_e = 0
            for frag in self.two_body:
                if use_frag_energies:
                    tbt_e += self._filter_frag_energy(frag, desired_occs=desired_occs)
                elif diag_complete_space:
                    tbt_e += self._diagonalize_operator_manual_ss(frag.to_op())
                else:
                    tbt_e += self._diagonalize_operator_with_ss_proj(frag.to_op())
            return const_obt + tbt_e
        elif not self.partitioned and not self.fluid:
            if not (
                isinstance(self.one_body, np.ndarray)
                and isinstance(self.two_body, np.ndarray)
            ):
                raise UserWarning(
                    "Expected one-electron and two-electron parts to be tensors."
                )
            return self._diagonalize_operator_with_ss_proj(
                self.constant + obt2op(self.one_body) + tbt2op(self.two_body)
            )
        else:
            raise UserWarning("Shouldn't end up here.")

    def save(self, id: str = ""):
        file_name = os.path.join(
            self.m_config.f3_folder, self.m_config.mol_name, self.m_config.date
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
        file_name += f"_{id}" if len(id) > 0 else ""
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


from min_part.f3_optimis import (
    ofat_fluid_optimize,
    afao_fluid_optimize,
    greedy_coeff_optimize,
    convex_optimization,
)  # noqa: E402
