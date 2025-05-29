import unittest
from functools import reduce

import numpy as np
from openfermion import (
    count_qubits,
)

from min_part.ffrag_utils import get_u_from_angles
from min_part.ham_decomp import (
    make_supermatrix,
    four_tensor_to_two_tensor_indices,
    make_x_matrix,
    make_unitary,
    make_fr_tensor,
    gfr_cost,
    frob_norm,
    gfro_decomp,
)
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op
from pert_trotter.fermi_frag import Do_GFRO


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

    # === Low Rank Helpers ===
    def test_4_to_2_indices(self):
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 4, n=5)
        self.assertEqual(pq, 0)
        self.assertEqual(rs, 5)

    def test_supermatrix_2(self):
        test_matrix = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
            ]
        )
        supermatrix = make_supermatrix(test_matrix)
        self.assertEqual(test_matrix[0][0][1][1], supermatrix[0][3])
        print(test_matrix * test_matrix)

    def test_supermatrix_4(self):
        test_matrix = np.random.rand(4, 4, 4, 4)
        supermatrix = make_supermatrix(test_matrix)
        self.assertEqual(test_matrix[0][0][1][1], supermatrix[0][5])
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 4, n=5)
        self.assertEqual(test_matrix[0][0][1][4], supermatrix[pq][rs])
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 4, n=5)
        self.assertEqual(test_matrix[0][0][1][4], supermatrix[pq][rs])

    # === Greedy Full Rank Helpers ===
    def test_frob_norm(self):
        n = 5
        m = (n * (n + 1)) // 2
        lambdas = np.random.rand(m)
        thetas = np.random.rand(m)
        tensor = make_fr_tensor(lambdas, thetas, n)
        tensor_half = tensor - 0.5 * tensor
        larger_norm = frob_norm(tensor)
        smaller_norm = frob_norm(tensor_half)
        self.assertTrue(larger_norm > smaller_norm)

    def test_cost_function(self):
        n = 5
        m = (n * (n + 1)) // 2
        thetas = np.random.rand(m)
        lambdas = np.random.rand(m)
        tensor = make_fr_tensor(lambdas, thetas, n)
        res = gfr_cost(lambdas, thetas, tensor, n)
        self.assertEqual(res, 0)
        non_zero = gfr_cost(
            lambdas, thetas, make_fr_tensor(lambdas, np.random.rand(m), n), n
        )
        self.assertNotEqual(non_zero, 0)

    def test_make_X(self):
        n = 10
        m = (n * (n + 1)) // 2
        x = make_x_matrix(thetas=np.random.rand(m), n=10)
        self.assertEqual(x[8][9], -x[9][8])
        self.assertEqual(x[4][5], -x[5][4])
        self.assertEqual(x[3][7], -x[7][3])

    def test_make_U(self):
        n = 4
        thetas = np.array([0, 1, 2, 3, 0, 4, 5, 0, 6, 0])
        try:
            u = make_unitary(thetas, n)
        except:
            self.fail()

    def test_make_fr_tensor(self):
        n = 5
        m = (n * (n + 1)) // 2
        thetas = np.random.rand(m)
        lambdas = np.random.rand(m)
        try:
            tensor = make_fr_tensor(lambdas, thetas, n)
            fo = tbt2op(tensor)
        except:
            self.fail()

    def test_grfo(self):
        """This test checks for the correct GFRO partitioning of H2.

        Some constraints to be checked are:

        The sum of the GFRO fragments == the sum of the unpartitioned fragments
        Each U at each step chosen are unitary

        """
        gfro_frags = gfro_decomp(tbt=self.H_tbt)
        n = self.H_tbt.shape[0]
        for frag in gfro_frags:
            u = make_unitary(frag.thetas, n)
            self.assertAlmostEqual(np.linalg.det(u), 1, places=5)

        self.assertEqual(
            reduce(lambda op1, op2: op1 + op2, [f.operators for f in gfro_frags]),
            self.H_tb_op,
        )

    def test_other_gfro(self):
        all_frag_ops, gfro_fragments, gfro_params = Do_GFRO(
            self.H_ele, shrink_frag=False, CISD=False
        )
        for coeffs, angles in gfro_params:
            u = get_u_from_angles(angles, 4)
            self.assertAlmostEqual(np.linalg.det(u), 1, places=5)

        self.assertEqual(
            reduce(lambda op1, op2: op1 + op2, gfro_fragments),
            self.H_tb_op,
        )
