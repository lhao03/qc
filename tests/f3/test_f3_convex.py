import unittest

import numpy as np
from openfermion import random_hermitian_matrix, eigenspectrum
from opt_einsum import contract

from d_types.cvx_exp import (
    make_fluid_variables,
    make_ob_matrices,
    get_energy_expressions,
)
from d_types.fragment_types import (
    OneBodyFragment,
    Subspace,
    PartitionStrategy,
    ContractPattern,
)
from d_types.hamiltonian import FragmentedHamiltonian, OptType
from min_part.f3_opers import obt2fluid, make_unitary_jl
from min_part.f3_optimis import simple_convex_opt
from min_part.molecules import h2_settings
from min_part.tensor import obt2op, extract_thetas, tbt2op
from tests.f3.test_ham_obj import get_tensors


class F3ConvexTest(unittest.TestCase):
    def test_frag_energy_summation(
        self, m_config=h2_settings, bond_length: float = 0.8
    ):
        const, obt, tbt = get_tensors(h2_settings, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        for f in ham.two_body:
            f.to_fluid()

    def test_ob_convex_opt(self):
        def min_eig(t):
            return min(np.linalg.eigh(t)[0])

        _, obt, _ = get_tensors(h2_settings, 0.8)
        O: OneBodyFragment = obt2fluid(obt)
        A: OneBodyFragment = obt2fluid(random_hermitian_matrix(n=4, real=True))
        B: OneBodyFragment = obt2fluid(random_hermitian_matrix(n=4, real=True))
        C: np.ndarray = O.to_tensor() + A.to_tensor() + B.to_tensor()
        print(f"spectrum: {eigenspectrum(operator=obt2op(C))}")
        print(
            f"sum of eigenvalues before optimization: {sum([min(eigenspectrum(operator=o.to_op())) for o in [O, A, B]])}"
        )
        simple_convex_opt(ob=O, frags=[A, B], min_eig=min_eig)
        np.testing.assert_array_almost_equal(
            np.trace(C),
            np.trace(O.to_tensor()) + np.trace(A.to_tensor()) + np.trace(B.to_tensor()),
        )
        optimized_eigensum = sum(
            [min(eigenspectrum(operator=o.to_op())) for o in [O, A, B]]
        )
        print(f"sum of eigenvalues after optimization: {optimized_eigensum}")
        self.assertAlmostEqual(
            optimized_eigensum, min(eigenspectrum(operator=obt2op(C))), places=4
        )

    def test_fluid_matrix_exp(self, m_config=h2_settings, bond_length=0.8):
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        n = len(gfro.two_body)
        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        fluid_variables = make_fluid_variables(n=n, self=gfro)
        unitaries = [make_unitary_jl(n=4, self=f) for f in gfro.two_body]
        ob_fluid_matrices = make_ob_matrices(
            contract_pattern=ContractPattern.GFRO,
            fluid_lambdas=fluid_variables,
            self=gfro,
            unitaries=unitaries,
        )
        for i, f in enumerate(gfro.two_body):
            f.to_fluid()
            for j in range(n):
                fluid_variables[(i * n) + j].value = gfro.two_body[
                    i
                ].fluid_parts.fluid_lambdas[j]
        for i, m in enumerate(ob_fluid_matrices):
            np_contract = contract(
                ContractPattern.GFRO.value,
                gfro.two_body[i].fluid_parts.fluid_lambdas,
                unitaries[i],
                unitaries[i],
            )
            cv_mat = m.value
            np.testing.assert_array_almost_equal(np_contract, cv_mat)

    def test_frag_energies(self, m_config=h2_settings, bond_length=0.8):
        const, obt, tbt = get_tensors(m_config, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        n = 4
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        fluid_variables = make_fluid_variables(n=n, self=ham)
        num_coeffs = []
        for f in ham.two_body:
            f.to_fluid()
            num_coeffs += list(f.fluid_parts.fluid_lambdas)
        energy_expressions = get_energy_expressions(
            i=0,
            n=n,
            num_coeffs=num_coeffs,
            f=ham.two_body[0],
            fluid_variables=fluid_variables,
            desired_occs=[
                (0,),
                (0, 1),
                (0, 1, 2),
                (0, 1, 2, 3),
            ],
        )
        for i, f in enumerate(ham.two_body):
            f.to_fluid()
            for j in range(n):
                c = ham.two_body[i].fluid_parts.fluid_lambdas[j] / 2
                fluid_variables[(i * n) + j].value = c
                f.move2frag(to=ham.one_body, coeff=c, orb=j, mutate=True)
        occs1, frag_energies_1 = ham.two_body[0].get_expectation_value(
            num_spin_orbs=4, expected_e=1
        )
        for i, occ in enumerate(occs1):
            if occ == (0,):
                self.assertEqual(frag_energies_1[i], energy_expressions[0].value)
                break
        occs2, frag_energies_2 = ham.two_body[0].get_expectation_value(
            num_spin_orbs=4, expected_e=2
        )
        for i, occ in enumerate(occs2):
            if occ == (0, 1):
                self.assertAlmostEqual(frag_energies_2[i], energy_expressions[1].value)
                break
        occs3, frag_energies_3 = ham.two_body[0].get_expectation_value(
            num_spin_orbs=4, expected_e=3
        )
        for i, occ in enumerate(occs3):
            if occ == (0, 1, 2):
                self.assertAlmostEqual(frag_energies_3[i], energy_expressions[2].value)
                break
        occs4, frag_energies_4 = ham.two_body[0].get_expectation_value(
            num_spin_orbs=4, expected_e=4
        )
        for i, occ in enumerate(occs4):
            if occ == (0, 1, 2, 3):
                self.assertAlmostEqual(frag_energies_4[i], energy_expressions[3].value)
                break

    def test_simple_ob_tb(self, m_config=h2_settings, bond_length=0.8):
        const, obt, tbt = get_tensors(m_config, bond_length)
        random_hermitian = random_hermitian_matrix(n=4, real=True, seed=0)
        unitary = np.diagflat(np.ones(4))
        tb_tensor = contract(
            "lm,lp,lq,mr,ms->pqrs", random_hermitian, unitary, unitary, unitary, unitary
        )
        tb_op = tbt2op(tb_tensor)
        thetas, _ = extract_thetas(unitary)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        ham.two_body = [ham.two_body[0]]
        total_op = obt2op(obt) + ham.two_body[0].to_op()
        print(f"total eigenspectrum: {eigenspectrum(total_op)}")
        print(
            f"starting energy: {ham.get_expectation_value(use_frag_energies=True, desired_occs=[(0, 1), (1, 2), (2, 3)])}"
        )
        print(
            f"summed eigenspectrum (over complete hilbert space): {sum([min(eigenspectrum(o)) for o in [obt2op(obt), ham.two_body[0].to_op()]])}"
        )
        desired_occs = [(0, 1), (1, 2), (2, 3)]
        ham.optimize_fragments(
            optimization_type=OptType.CONVEX, desired_occs=[(0, 1), (1, 2), (2, 3)]
        )
        vals, vecs = np.linalg.eigh(ham.one_body.to_tensor())
        print(vals)
        tbt_e = 0
        for f in ham.two_body:
            es = []
            occs, e = f.get_expectation_value(4, 2)
            for i, occ in enumerate(occs):
                if occ in desired_occs:
                    es.append(e[i])
            tbt_e += min(es)
        print(tbt_e)
