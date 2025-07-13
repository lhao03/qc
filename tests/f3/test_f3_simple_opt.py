import json
import os
import unittest

import numpy as np
from openfermion import FermionOperator

from d_types.config_types import PartitionStrategy, Subspace
from d_types.hamiltonian import FragmentedHamiltonian, OptType
from min_part.f3_opers import obt2fluid
from min_part.f3_optimis import greedy_E_optimize, simple_convex_opt
from min_part.julia_ops import jl_print
from min_part.molecules import h2_settings
from min_part.operators import (
    subspace_projection_operator,
)
from min_part.plots import plot_energies, FluidPlotNames
from min_part.tensor import obt2op
from tests.utils.sim_tensor import get_tensors


class F3OptTest(unittest.TestCase):
    def test_simple_matrix_opt(self):
        def random_mutate(obt):
            return obt - 2.3, obt + 2.3

        def min_eig(t):
            return min(np.linalg.eigh(t)[0])

        _, obt, _ = get_tensors(h2_settings, 0.6)
        obt_A, obt_mut = random_mutate(obt)
        _, obt_O, _ = get_tensors(h2_settings, 2.3)
        O = obt2fluid(obt_O)
        B = obt2fluid(obt_mut)
        A = obt2fluid(obt_A)
        C = obt_mut + obt_A + obt_O
        print(f"eigs before transform:{np.linalg.eigh(C)[0]}")
        print(
            f"Partitioned min lambda: {min_eig(B.to_tensor()) + min_eig(A.to_tensor()) + min_eig(O.to_tensor())}"
        )
        print("Before optimization")
        jl_print(B.to_tensor())
        jl_print(A.to_tensor())
        jl_print(O.to_tensor())
        print("==")
        scipy_run = False
        if scipy_run:
            greedy_E_optimize(ob=O, frags=[B, A], iters=1000, min_eig=min_eig)
        else:
            simple_convex_opt(ob=O, frags=[A, B], min_eig=min_eig)
        print(
            f"Optimized Partitioned E: {min_eig(B.to_tensor()) + min_eig(A.to_tensor()) + min_eig(O.to_tensor())}"
        )
        self.assertAlmostEqual(
            np.trace(C),
            np.trace(B.to_tensor()) + np.trace(A.to_tensor()) + np.trace(O.to_tensor()),
        )
        np.testing.assert_array_almost_equal(
            C, B.to_tensor() + A.to_tensor() + O.to_tensor()
        )
        print("After optimization")
        jl_print(B.to_tensor())
        jl_print(A.to_tensor())
        jl_print(O.to_tensor())
        print("==")

    def test_simple_jw_opt(self):
        def random_mutate(obt):
            return obt - 2.3, obt + 2.3

        def min_eig(fo):
            if not isinstance(fo, FermionOperator):
                fo = obt2op(fo)
            arr = subspace_projection_operator(fo, 4, 2).toarray()
            return min(np.linalg.eigh(arr)[0])

        _, obt, _ = get_tensors(h2_settings, 0.6)
        obt_A, obt_mut = random_mutate(obt)
        _, obt_O, _ = get_tensors(h2_settings, 2.3)
        O = obt2fluid(obt_O)
        B = obt2fluid(obt_mut)
        A = obt2fluid(obt_A)
        C = obt_mut + obt_A + obt_O
        print(f"eigs before transform:{min_eig(C)}")
        print(
            f"Partitioned min lambda: {min_eig(B.to_tensor()) + min_eig(A.to_tensor()) + min_eig(O.to_tensor())}"
        )
        print("Before optimization")
        jl_print(B.to_tensor())
        jl_print(A.to_tensor())
        jl_print(O.to_tensor())
        print("==")
        scipy_run = True
        if scipy_run:
            greedy_E_optimize(ob=O, frags=[A, B], iters=100, min_eig=min_eig)
        else:
            simple_convex_opt(ob=B, frags=[O, A], min_eig=min_eig)
        print(
            f"Optimized Partitioned E: {min_eig(B.to_tensor()) + min_eig(A.to_tensor()) + min_eig(O.to_tensor())}"
        )
        self.assertAlmostEqual(
            np.trace(C),
            np.trace(B.to_tensor()) + np.trace(A.to_tensor()) + np.trace(O.to_tensor()),
        )
        np.testing.assert_array_almost_equal(
            C, B.to_tensor() + A.to_tensor() + O.to_tensor()
        )
        print("After optimization")
        jl_print(B.to_tensor())
        jl_print(A.to_tensor())
        jl_print(O.to_tensor())
        print("==")

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

        def min_eig(fo):
            if not isinstance(fo, FermionOperator):
                fo = obt2op(fo)
            arr = subspace_projection_operator(fo, 4, 2).toarray()
            return min(np.linalg.eigh(arr)[0])

        ham.optimize_fragments(
            iters=1000, debug=True, min_eig=min_eig, optimization_type=OptType.CONVEX
        )
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
