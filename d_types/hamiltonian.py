import os
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional, Tuple

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

from d_types.config_types import MConfig, PartitionStrategy
from d_types.cvx_exp import make_fluid_variables
from d_types.fragment_types import GFROFragment, OneBodyFragment, FermionicFragment
from min_part.f3_opers import obt2fluid

from min_part.gfro_decomp import gfro_decomp
from min_part.lr_decomp import lr_decomp
from min_part.operators import (
    subspace_restriction,
    generate_occupied_spin_orb_permutations,
)
from min_part.plots import RefLBPlotNames, plot_energies
from min_part.tensor import obt2op, tbt2op
from min_part.utils import open_frags, save_frags
from tests.utils.sim_tensor import get_tensors


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
    restrictor: Optional[partial] = None
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
            return (
                config_eq
                and constant_eq
                and ob_eq
                and tb_eq
                and partitioned_eq
                and fluid_eq
            )
        else:
            return False

    def _return_proper_fluid_vars(self, frag):
        n = self.m_config.num_spin_orbs
        for i, f in enumerate(self.two_body):
            if f == frag:
                return self.fluid_variables[i * n, (i * n) + n]

    def optimize_fragments(self, optimization_type: OptType, filter_sz: bool = False):
        self.fluid_variables = make_fluid_variables(
            n=self.one_body.lambdas.shape[0], self=self
        )
        desired_occs = generate_occupied_spin_orb_permutations(
            total_spin_orbs=self.m_config.num_spin_orbs, occ=self.m_config.gs_elecs
        )
        if filter_sz:
            desired_occs = list(filter(zero_s_z, desired_occs))
        match optimization_type:
            case OptType.CONVEX:
                return convex_optimization(self, desired_occs)

    def _add_up_orb_occs(self, frag: FermionicFragment):
        occs, energies = frag.get_expectation_value(
            num_spin_orbs=self.m_config.num_spin_orbs,
            expected_e=self.m_config.gs_elecs,
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
                subspace_restriction(
                    fo, self.m_config.num_spin_orbs, self.m_config.gs_elecs
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
                (num_elecs == self.m_config.gs_elecs)
                and (np.isclose(self.m_config.s2, s2))
                and (np.isclose(self.m_config.sz, sz))
            ):
                energies.append(eigs[i])
        return min(energies)

    def _diagonalize_operator_with_ss_proj(self, fo: FermionOperator):
        eigenvalues, eigenvectors = np.linalg.eigh(self.restrictor(fo).toarray())
        return min(eigenvalues, default=0)

    def _filter_frag_energy(
        self, frag: FermionicFragment, desired_occs: List[Tuple] = None
    ):
        occs, energies = frag.get_expectation_value(
            num_spin_orbs=self.m_config.num_spin_orbs, expected_e=self.m_config.gs_elecs
        )
        if desired_occs is not None:
            desired_energies = []
            for occ, energy in zip(occs, energies):
                if occ in desired_occs:
                    desired_energies.append(energy)
            return min(desired_energies)
        else:
            return min(energies, default=0)

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
        self,
        strategy: PartitionStrategy,
        bond_length: float,
        load_prev: bool = True,
        save: bool = True,
    ):
        frag_path = self.m_config.get_unique_frag_folder(
            frag_type="gfro" if strategy is PartitionStrategy.GFRO else "lr",
            bond_length=str(bond_length),
        )

        if load_prev and os.path.exists(f"{frag_path}.pkl"):
            self.two_body = open_frags(frag_path)
        else:
            match strategy:
                case PartitionStrategy.GFRO:
                    self.two_body = gfro_decomp(self.two_body, debug=True)
                case PartitionStrategy.LR:
                    self.two_body = lr_decomp(self.two_body)
            if save:
                save_frags(self.two_body, file_name=frag_path)
        self.partitioned = True
        self.one_body = obt2fluid(self.one_body)
        return self.two_body

    def get_expectation_value(
        self, use_frag_energies: bool = False, filter_sz: bool = True
    ):
        if not self.restrictor:
            self.restrictor = partial(
                subspace_restriction,
                n_spin_orbs=self.m_config.num_spin_orbs,
                num_elecs=self.m_config.gs_elecs,
                ci_projection=self.ci_projection,
            )
        if self.partitioned or self.fluid:
            const_obt = self._diagonalize_operator_with_ss_proj(
                self.constant + self.one_body.to_op()
            )
            tbt_e = 0
            for frag in self.two_body:
                if use_frag_energies:
                    desired_occs = generate_occupied_spin_orb_permutations(
                        total_spin_orbs=self.m_config.num_spin_orbs,
                        occ=self.m_config.gs_elecs,
                    )
                    if filter_sz:
                        desired_occs = list(filter(zero_s_z, desired_occs))
                    tbt_e += self._filter_frag_energy(frag, desired_occs=desired_occs)
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

    @classmethod
    def generate_curves(
        cls,
        m_config: MConfig,
        use_frag_energies: bool = True,
        filter_sz: bool = True,
        partition_strat: PartitionStrategy = PartitionStrategy.GFRO,
        exact: bool = True,
        sep_one_two: bool = True,
        fragment: bool = False,
        f3_opt: bool = False,
    ):
        energies = []
        labels = []
        exact_e = []
        sep_e = []
        frag_e = []
        f3_e = []
        for b in m_config.xpoints:
            const, obt, tbt = get_tensors(m_config, b)
            temp_ham = FragmentedHamiltonian(
                m_config=m_config,
                constant=const,
                one_body=obt,
                two_body=tbt,
                partitioned=False,
                fluid=False,
            )
            if exact:
                exact_e.append(
                    temp_ham.get_expectation_value(
                        use_frag_energies=use_frag_energies, filter_sz=filter_sz
                    )
                )
                energies.append(exact_e)
                labels.append(RefLBPlotNames.NO_PARTITIONING)
            if sep_one_two:
                sep_e.append(
                    temp_ham._diagonalize_operator_with_ss_proj(
                        temp_ham.constant + obt2op(temp_ham.one_body)
                    )
                    + temp_ham._diagonalize_operator_with_ss_proj(
                        tbt2op(temp_ham.two_body)
                    )
                )
            if fragment:
                temp_ham.partition(strategy=partition_strat, bond_length=b)
                energies.append(frag_e)
                labels.append(
                    RefLBPlotNames.GFRO
                    if partition_strat is PartitionStrategy.GFRO
                    else RefLBPlotNames.LR
                )
            if f3_opt:
                temp_ham.optimize_fragments(
                    optimization_type=OptType.CONVEX, filter_sz=filter_sz
                )
                energies.append(f3_e)
                labels.append(
                    RefLBPlotNames.F3_GFRO
                    if partition_strat is PartitionStrategy.GFRO
                    else RefLBPlotNames.F3_LR
                )
        return energies, labels

    @classmethod
    def plot_curves(
        cls, m_config: MConfig, title: str, energies: List[List[float]], labels: List
    ):
        plot_energies(
            xpoints=m_config.xpoints,
            points=energies,
            title=f"{m_config.mol_name} {title}",
            labels=labels,
            dir=m_config.results_folder,
        )


from min_part.f3_optimis import (  # noqa: E402
    convex_optimization,
)
