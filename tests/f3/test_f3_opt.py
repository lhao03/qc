import json
import os
import unittest

import numpy as np
from openfermion import qubit_operator_sparse, jordan_wigner

from d_types.fragment_types import PartitionStrategy, Subspace
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_opers import obt2fluid
from min_part.f3_optimis import greedy_E_optimize
from min_part.molecules import h2_settings
from min_part.operators import (
    subspace_projection_operator,
    get_particle_number,
    get_total_spin,
    get_projected_spin,
)
from min_part.plots import plot_energies, FluidPlotNames
from min_part.tensor import obt2op
from tests.f3.test_ham_obj import get_tensors


class F3OptTest(unittest.TestCase):
    def test_simple_E_opt(self):
        def random_mutate(obt):
            return obt - 2.3, obt + 2.3

        def get_min_eig(fo):
            return min(
                np.linalg.eigh(subspace_projection_operator(fo, 4, 2).toarray())[0]
            )

        _, obt, _ = get_tensors(h2_settings, 0.6)
        obt_A, obt_mut = random_mutate(obt)
        _, obt_B, _ = get_tensors(h2_settings, 2.3)
        O = obt2fluid(obt_mut)
        A = obt2fluid(obt_A)
        B = obt2fluid(obt_B)
        C = obt_mut + obt_A + obt_B
        filtered_min_eig = []
        vals, vecs = np.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(obt2op(C))).toarray()
        )
        for i in range(len(vals)):
            n = get_particle_number(vecs[:, i], 4)
            s2 = get_total_spin(vecs[:, i], 2)
            sz = get_projected_spin(vecs[:, i], 2)
            if n == 2 and s2 == 0 and sz == 0:
                filtered_min_eig.append(vals[i])
        ss_min_eig = get_min_eig(obt2op(C))
        filtered_min_eig = min(filtered_min_eig)
        self.assertAlmostEqual(ss_min_eig, filtered_min_eig)
        abs_min = min(vals)
        abs_max = max(vals)
        print(f"Absolute min: {abs_min}, Absolute max: {abs_max}")
        partioned_E = (
            get_min_eig(O.to_op()) + get_min_eig(A.to_op()) + get_min_eig(B.to_op())
        )
        print(f"Exact E: {ss_min_eig}")
        print(f"Partioned E: {partioned_E}")
        print(
            f"Partioned E (solution): {get_min_eig(obt2op(obt)) + get_min_eig(obt2op(obt)) + get_min_eig(B.to_op())}"
        )
        greedy_E_optimize(ob=O, frags=[A, B], iters=1000, debug=True)
        partioned_E = (
            get_min_eig(O.to_op()) + get_min_eig(A.to_op()) + get_min_eig(B.to_op())
        )
        print(f"Optimized Partioned E: {partioned_E}")
        self.assertAlmostEqual(
            np.trace(C),
            np.trace(O.to_tensor()) + np.trace(A.to_tensor()) + np.trace(B.to_tensor()),
        )

    def test_gfro_opt(self, bond_length=0.8, m_config=h2_settings):
        print(f"Partitioning bond {bond_length}")
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(expected_e=2, expected_sz=0, expected_s2=0),
        )
        E = gfro.get_expectation_value()
        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        gfro_E = gfro.get_expectation_value()
        gfro.optimize_fragments()
        gfro_fluid_E = gfro.get_expectation_value()
        gfro.save(id=str(bond_length))
        print(f"Exact: {E}")
        print(f"Only GFRO Partitioning: {gfro_E}")
        print(f"Fluid GFRO: {gfro_fluid_E}")
        # self.assertTrue(E >= gfro_fluid_E >= gfro_E)
        return E, gfro_fluid_E, gfro_E

    def test_fluid_gfro(self):
        child_dir = os.path.join(
            "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/data",
            h2_settings.date,
        )
        no_partitioning = []
        gfro_fluid = []
        gfro = []
        for bond_length in h2_settings.xpoints:
            E, gfro_fluid_E, gfro_E = self.test_gfro_opt(
                bond_length=bond_length, m_config=h2_settings
            )
            print(f"E: {E}, Fluid GFRO E: {gfro_fluid_E}, GFRO E: {gfro_E}")
            no_partitioning.append(E)
            gfro_fluid.append(gfro_fluid_E)
            gfro.append(gfro_E)
        plot_energies(
            xpoints=h2_settings.xpoints,
            points=[no_partitioning, gfro_fluid, gfro],
            title=f"GFRO Lower Bounds for {h2_settings.mol_name}",
            labels=[
                FluidPlotNames.NO_PARTITIONING,
                FluidPlotNames.GFRO_FLUID,
                FluidPlotNames.GFRO,
            ],
            dir=child_dir,
        )
        energies = {
            FluidPlotNames.NO_PARTITIONING.value: no_partitioning,
            FluidPlotNames.GFRO_FLUID.value: gfro_fluid,
            FluidPlotNames.GFRO.value: gfro,
        }

        energies_json = json.dumps(energies)
        with open(
            os.path.join(child_dir, f"{h2_settings.mol_name}.json"),
            "w",
        ) as f:
            f.write(energies_json)
