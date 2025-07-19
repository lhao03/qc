import random
import time
import unittest
from functools import reduce

import numpy as np
import ray
import scipy as sp
from openfermion import (
    jordan_wigner,
    qubit_operator_sparse,
)

from d_types.config_types import PartitionStrategy, Basis
from d_types.hamiltonian import FragmentedHamiltonian
from d_types.unitary_type import make_x_matrix, make_unitary, Unitary
from min_part.gfro_decomp import (
    gfro_cost,
    frob_norm,
    gfro_decomp,
    make_fr_tensor,
)
from d_types.fragment_types import gfro_fragment_occ, GFROFragment
from min_part.molecules import h2_settings, h2o_settings, h4_settings
from min_part.operators import (
    generate_occupied_spin_orb_permutations,
)
from min_part.tensor import (
    get_no_from_tensor,
    tbt2op,
    make_lambda_matrix,
    make_fr_tensor_from_u,
    spin2spac,
)
from tests.utils.sim_tensor import get_tensors


class GFROTest(unittest.TestCase):
    def test_frob_norm(self):
        n = 5
        m = (n * (n + 1)) // 2
        lambdas = np.random.rand(m)
        thetas = np.random.rand(m - n)
        tensor = make_fr_tensor(lambdas, thetas, n)
        tensor_half = tensor - 0.5 * tensor
        larger_norm = frob_norm(tensor)
        smaller_norm = frob_norm(tensor_half)
        self.assertTrue(larger_norm > smaller_norm)

    def test_cost_function(self):
        n = 5
        m = (n * (n + 1)) // 2
        thetas = np.random.rand(m - n)
        lambdas = np.random.rand(m)
        tensor = make_fr_tensor(lambdas, thetas, n)
        res = gfro_cost(lambdas, thetas, tensor, n)
        self.assertEqual(res, 0)
        non_zero = gfro_cost(
            lambdas,
            thetas,
            make_fr_tensor(lambdas, np.random.rand(m - n), n),
            n,
        )
        self.assertNotEqual(non_zero, 0)

    def test_make_X(self):
        n = 10
        m = (n * (n + 1)) // 2 - n
        x = make_x_matrix(thetas=np.random.rand(m), n=10)
        self.assertEqual(x[8][9], -x[9][8])
        self.assertEqual(x[4][5], -x[5][4])
        self.assertEqual(x[3][7], -x[7][3])

    def test_make_permutations(self):
        n = 3
        combinations_s = generate_occupied_spin_orb_permutations(n, None)
        self.assertEqual(
            [
                (),
                (0,),
                (1,),
                (2,),
                (0, 1),
                (0, 2),
                (1, 2),
                (
                    0,
                    1,
                    2,
                ),
            ],
            combinations_s,
        )

    def test_make_fr_tensor_two_way(self):
        n = 4
        m = (n * (n + 1)) // 2
        thetas = np.random.rand(m - n)
        lambdas = np.random.rand(m)
        u = make_unitary(thetas, n)
        tensor_from_lambdas_thetas = make_fr_tensor(lambdas=lambdas, thetas=thetas, n=n)
        tensor_from_lambdas_u = make_fr_tensor_from_u(lambdas=lambdas, u=u, n=n)
        np.testing.assert_array_equal(tensor_from_lambdas_thetas, tensor_from_lambdas_u)

    def test_get_no_from_tensor(self):
        n = 4
        m = n * (n + 1) // 2
        only_diag_lambdas = np.zeros(m)
        only_diag_lambdas[0] = random.randint(1, 10) * 0.1
        only_diag_lambdas[4] = random.randint(1, 10) * 0.1
        only_diag_lambdas[7] = random.randint(1, 10) * 0.1
        only_diag_lambdas[9] = random.randint(1, 10) * 0.1
        diag_matrix_lm = make_lambda_matrix(only_diag_lambdas, n)
        only_diag_operators = get_no_from_tensor(diag_matrix_lm)
        self.assertEqual(n, len(only_diag_operators.terms))
        for term, ceoff in only_diag_operators.terms.items():
            i = term[0][0]
            self.assertEqual(diag_matrix_lm[i][i], ceoff)

        full_lambdas = np.random.rand(m)
        full_lm = make_lambda_matrix(full_lambdas, n)
        full_lambda_operator = get_no_from_tensor(full_lm)
        self.assertEqual(n * n, len(full_lambda_operator.terms))
        for term, coeff in full_lambda_operator.terms.items():
            l = term[0][0]
            m = term[2][0]
            self.assertEqual(coeff, full_lm[l][m])

    def test_grfo_h2(self):
        const, obt, tbt = get_tensors(h2_settings, 0.8)
        gfro_frags = gfro_decomp(tbt=tbt, debug=True)
        for frag in gfro_frags:
            u = frag.unitary.make_unitary_matrix()
            np.testing.assert_array_almost_equal(
                u, make_unitary(frag.unitary.thetas, frag.unitary.dim)
            )
            self.assertAlmostEqual(np.linalg.det(u), 1, places=7)

        self.assertEqual(
            reduce(lambda op1, op2: op1 + op2, [f.operators for f in gfro_frags]),
            tbt2op(tbt),
        )

    def test_grfo_artificial(self):
        n = 4
        m = n * (n + 1) // 2
        fake_u = np.array(
            [
                [0.70710029, 0.00303002, 0.70710028, 0.00303002],
                [-0.00303002, 0.70710029, -0.00303002, 0.70710028],
                [-0.70710493, -0.00161596, 0.70710494, 0.00161596],
                [0.00161597, -0.70710493, -0.00161596, 0.70710494],
            ]
        )
        fake_lambdas = np.array(sorted([0.1 * random.randint(1, 10) for _ in range(m)]))
        self.assertAlmostEqual(np.linalg.det(fake_u), 1, places=7)
        fake_hamiltonian = make_fr_tensor_from_u(fake_lambdas, fake_u, n)
        gfro_frags = gfro_decomp(fake_hamiltonian)
        self.assertTrue(len(gfro_frags) == 1)
        self.assertEqual(gfro_frags[0].operators, tbt2op(fake_hamiltonian))
        np.testing.assert_array_almost_equal(
            sorted(gfro_frags[0].lambdas), fake_lambdas
        )

        fr_u = gfro_frags[0].unitary.make_unitary_matrix()
        rows_checked = 0
        for row in fake_u:
            for i, gen_row in enumerate(fr_u):
                try:
                    if np.allclose(row, gen_row, rtol=1e-05, atol=1e-07) or np.allclose(
                        row, -1 * gen_row, rtol=1e-05, atol=1e-07
                    ):
                        fr_u = np.delete(fr_u, (i), axis=0)
                        rows_checked += 1
                        break
                except:
                    continue
        self.assertEqual(rows_checked, fake_u.shape[0])

    def test_grfo_artificial_occs(self):
        """This test checks that diagonalization of GFRO fragments returns
        the same values as substituting in occupation numbers.
        """
        n = 4
        m = n * (n + 1) // 2
        fake_u = np.array(
            [
                [0.70710029, 0.00303002, 0.70710028, 0.00303002],
                [-0.00303002, 0.70710029, -0.00303002, 0.70710028],
                [-0.70710493, -0.00161596, 0.70710494, 0.00161596],
                [0.00161597, -0.70710493, -0.00161596, 0.70710494],
            ]
        )
        fake_lambdas = np.array(sorted([0.1 * random.randint(1, 10) for _ in range(m)]))
        fake_hamiltonian = make_fr_tensor_from_u(fake_lambdas, fake_u, n)
        fake_hamiltonian_operator = tbt2op(fake_hamiltonian)
        gfro_frags = gfro_decomp(fake_hamiltonian)
        self.assertEqual(1, len(gfro_frags))
        for frag_details in gfro_frags:
            diag_eigenvalues, diag_eigenvectors = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(frag_details.operators)).toarray()
            )
            occupations, eigenvalues = gfro_fragment_occ(
                fragment=frag_details, num_spin_orbs=n, occ=None
            )
            np.testing.assert_array_almost_equal(
                np.sort(diag_eigenvalues), np.sort(eigenvalues)
            )
            self.assertEqual(fake_hamiltonian_operator, frag_details.operators)

    def test_spac2spin_simple(self):
        spac_frag = GFROFragment(
            basis=Basis.SPATIAL,
            unitary=Unitary.deconstruct_unitary(np.identity(2), basis=Basis.SPATIAL),
            lambdas=np.random.rand(3),
        )
        self.assertEqual(spac_frag, spac_frag.spac2spin().spin2spac())

    def test_spac2spin_h2(self):
        frags = gfro_decomp(get_tensors(h2_settings, 0.8)[2], basis=Basis.SPIN)
        og_spin_frag = frags[0]
        spat_tbt = spin2spac(og_spin_frag.to_tensor())
        spatial_frag = gfro_decomp(spat_tbt, basis=Basis.SPATIAL)[0]
        spin_frag = spatial_frag.spac2spin()
        self.assertEqual(
            tbt2op(og_spin_frag.to_tensor()), tbt2op(spin_frag.to_tensor())
        )

    def test_spac2spin_many(self):
        _, _, h2_tbt = get_tensors(h2_settings, h2_settings.stable_bond_length)
        spac_h2_tbt = spin2spac(h2_tbt)
        spac_h2_frags = gfro_decomp(spac_h2_tbt, debug=True, basis=Basis.SPATIAL)
        spin_h2_frags = [f.spac2spin() for f in spac_h2_frags]
        sum_h2_op = sum([tbt2op(f.to_tensor()) for f in spin_h2_frags])
        self.assertEqual(tbt2op(h2_tbt), sum_h2_op)
        _, _, h4_tbt = get_tensors(h4_settings, h4_settings.stable_bond_length)
        _, _, h2o_tbt = get_tensors(h2o_settings, h2o_settings.stable_bond_length)

    def test_multi_partition_gfro(self):
        from min_part.remote import partition_frags

        num_cpus = 4
        ray.init(num_cpus=num_cpus)
        m_config = h2o_settings
        futures = [
            partition_frags.remote(b, m_config, PartitionStrategy.GFRO)
            for b in m_config.xpoints
        ]
        res = ray.get(futures)

    def test_water(self):
        start_p = time.time()
        m_config = h2o_settings
        bond_length = h2o_settings.stable_bond_length
        const, obt, tbt = get_tensors(m_config, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
        )
        ham.partition(
            strategy=PartitionStrategy.GFRO, bond_length=bond_length, save=True
        )
        end_p = time.time()
        print(f"finished partitioning: {bond_length} in {end_p - start_p}")
