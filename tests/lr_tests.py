import unittest

import numpy as np
from openfermion import (
    count_qubits,
)


from min_part.lr_decomp import make_supermatrix, four_tensor_to_two_tensor_indices
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class DecompTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.45
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
