import json
import os
import unittest

import numpy as np
from openfermion import qubit_operator_sparse, jordan_wigner

from d_types.fragment_types import PartitionStrategy, Subspace
from d_types.hamiltonian import FragmentedHamiltonian, OptType
from min_part.f3_opers import obt2fluid
from min_part.f3_optimis import simple_convex_opt
from min_part.julia_ops import jl_print
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
        partitioned_E = (
            get_min_eig(O.to_op()) + get_min_eig(A.to_op()) + get_min_eig(B.to_op())
        )
        print(f"Exact E: {ss_min_eig}")
        print(f"Partitioned E: {partitioned_E}")
        print(
            f"Partitioned E (solution): {get_min_eig(obt2op(obt)) + get_min_eig(obt2op(obt)) + get_min_eig(B.to_op())}"
        )
        print("Before optimization")
        jl_print(O.to_tensor())
        jl_print(A.to_tensor())
        jl_print(B.to_tensor())
        print("==")
        # greedy_E_optimize(ob=A, frags=[O, B], iters=10000, debug=True)
        simple_convex_opt(ob=A, frags=[O, B])
        partitioned_E = (
            get_min_eig(O.to_op()) + get_min_eig(A.to_op()) + get_min_eig(B.to_op())
        )
        print(f"Optimized Partioned E: {partitioned_E}")
        self.assertAlmostEqual(
            np.trace(C),
            np.trace(O.to_tensor()) + np.trace(A.to_tensor()) + np.trace(B.to_tensor()),
        )
        np.testing.assert_array_almost_equal(
            C, O.to_tensor() + A.to_tensor() + B.to_tensor()
        )
        print("Solution to partitioned matrices")
        jl_print(O.to_tensor())
        jl_print(A.to_tensor())
        jl_print(B.to_tensor())
        print("==")

    def test_simple(self):
        import cvxpy as cp

        # Create two scalar optimization variables.
        x = cp.Variable()
        y = cp.Variable()

        # Create two constraints.
        constraints = [x + y == 1, x - y >= 1]

        # Form objective.
        obj = cp.Minimize((x - y) ** 2)

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

    def test_fluid_opt(self, bond_length=0.8, m_config=h2_settings):
        print(f"Partitioning bond {bond_length}")
        const, obt, tbt = get_tensors(m_config, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(expected_e=2, expected_sz=0, expected_s2=0),
        )
        E = ham.get_expectation_value()
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        partitioned_E = ham.get_expectation_value()
        ham.optimize_fragments(iters=10000, debug=True, optimization_type=OptType.OFAO)
        print(f"After optimization: {ham.get_expectation_value()}")
        ham.save(id=str(bond_length))
        print(f"Exact: {E}")
        print(f"Before optimization: {partitioned_E}")
        print(f"After optimization: {ham.get_expectation_value()}")
        self.assertTrue(E >= ham.get_expectation_value() >= partitioned_E)
        return E, ham.get_expectation_value(), partitioned_E

    def test_fluid(self):
        child_dir = os.path.join(
            "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/data",
            h2_settings.date,
        )
        no_partitioning = []
        fluid = []
        partitioned = []
        for bond_length in h2_settings.xpoints:
            E, fluid_E, partitioned_E = self.test_fluid_opt(
                bond_length=bond_length, m_config=h2_settings
            )
            print(f"E: {E}, Fluid E: {fluid_E}, Partitioned E: {partitioned_E}")
            no_partitioning.append(E)
            fluid.append(fluid_E)
            partitioned.append(partitioned_E)
        plot_energies(
            xpoints=h2_settings.xpoints,
            points=[no_partitioning, fluid, partitioned],
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
            FluidPlotNames.GFRO_FLUID.value: fluid,
            FluidPlotNames.GFRO.value: partitioned,
        }

        energies_json = json.dumps(energies)
        with open(
            os.path.join(child_dir, f"{h2_settings.mol_name}.json"),
            "w",
        ) as f:
            f.write(energies_json)
