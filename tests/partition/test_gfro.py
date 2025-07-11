import random
import unittest
from functools import reduce

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    number_operator,
    s_squared_operator,
    sz_operator,
)

from min_part.gfro_decomp import (
    gfro_cost,
    frob_norm,
    gfro_decomp,
    gfro_fragment_occ,
    make_fr_tensor,
)
from min_part.operators import (
    generate_occupied_spin_orb_permutations,
    get_particle_number,
    get_total_spin,
    get_projected_spin,
)
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor import (
    get_no_from_tensor,
    obt2op,
    tbt2op,
    make_x_matrix,
    make_unitary,
    make_lambda_matrix,
    extract_thetas,
    make_fr_tensor_from_u,
)
from tests.utils.sim_molecules import specific_lr_decomp
from tests.utils.sim_tensor import get_chem_tensors


class DecompTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.8
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op

    # === Greedy Full Rank Helpers ===
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
        self.assertTrue(
            np.array_equal(tensor_from_lambdas_thetas, tensor_from_lambdas_u)
        )

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
        """This test checks for the correct GFRO partitioning of H2.

        Some constraints to be checked are:

        The sum of the GFRO fragments == the sum of the unpartitioned fragments
        Each U at each step chosen are unitary

        """
        gfro_frags = gfro_decomp(tbt=self.H_tbt)
        n = self.H_tbt.shape[0]
        for frag in gfro_frags:
            u = make_unitary(frag.thetas, n)
            thetas, diags = extract_thetas(u)
            np.testing.assert_array_almost_equal(u, make_unitary(thetas, 4))
            self.assertAlmostEqual(np.linalg.det(u), 1, places=7)

        self.assertEqual(
            reduce(lambda op1, op2: op1 + op2, [f.operators for f in gfro_frags]),
            self.H_tb_op,
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
        self.assertTrue(
            np.allclose(
                sorted(gfro_frags[0].lambdas), fake_lambdas, rtol=1e-05, atol=1e-08
            )
        )
        fr_u = make_unitary(thetas=gfro_frags[0].thetas, n=n)
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
            self.assertTrue(
                np.allclose(np.sort(diag_eigenvalues), np.sort(eigenvalues))
            )
            self.assertEqual(fake_hamiltonian_operator, frag_details.operators)

    def test_grfo_h2_occs(self):
        from openfermion import qubit_operator_sparse, jordan_wigner

        H_obt, H_tbt, frags, bl = specific_lr_decomp(0.8)
        n = self.H_tbt.shape[0]
        for frag_details in frags:
            print("frag")
            diag_eigenvalues, diag_eigenvectors = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(frag_details.operators)).toarray()
            )

            n_op = qubit_operator_sparse(jordan_wigner(number_operator(4))).toarray()
            s2 = qubit_operator_sparse(jordan_wigner(s_squared_operator(2))).toarray()
            sz = qubit_operator_sparse(jordan_wigner(sz_operator(2))).toarray()
            for i in range(16):
                vec = diag_eigenvectors[:, i]
                if np.allclose(n_op @ vec, 2 * vec):
                    print(diag_eigenvalues[i])
                    print(np.allclose(s2 @ vec, 0 * vec))
                    print(np.allclose(sz @ vec, 0 * vec))
                    print(
                        get_particle_number(diag_eigenvectors[:, i], 4),
                        get_total_spin(diag_eigenvectors[:, i], 2),
                        get_projected_spin(diag_eigenvectors[:, i], 2),
                    )
