import random
import unittest
from functools import reduce

import numpy as np
from openfermion import (
    count_qubits,
)

from min_part.gfro_decomp import make_fr_tensor_from_u
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.lr_decomp import (
    make_supermatrix,
    four_tensor_to_two_tensor_indices,
    lr_decomp,
)
from min_part.molecules import mol_h2
from min_part.tensor import get_n_body_tensor
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op
from min_part.utils import do_lr_fo


class DecompTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.80
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
        self.assertEqual(rs, 9)

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
        n = 4
        test_matrix = np.random.rand(n, n, n, n)
        supermatrix = make_supermatrix(test_matrix)
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 1, n)
        self.assertEqual(test_matrix[0][0][1][1], supermatrix[pq][rs])
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 2, n=n)
        self.assertEqual(test_matrix[0][0][1][2], supermatrix[pq][rs])
        pq, rs = four_tensor_to_two_tensor_indices(3, 2, 1, 2, n=n)
        self.assertEqual(test_matrix[3][2][1][2], supermatrix[pq][rs])

    def test_lr_decomp_h2(self):
        pass

    def test_lr_decomp_fake(self):
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
        lr_frags_details = lr_decomp(fake_hamiltonian)
        lr_fo, lr_params = do_lr_fo(tbt2op(fake_hamiltonian))
        lr_ops_og = reduce(lambda a, b: a + b, lr_fo)
        for i, lr_frag in enumerate(lr_frags_details):
            pt_t = lr_params[i][2]
            self.assertTrue(
                np.allclose(get_n_body_tensor(lr_frag.operators, 2, 4), pt_t)
            )
        lr_operators = reduce(lambda a, b: a + b, [f.operators for f in lr_frags_details])
        og_op = tbt2op(fake_hamiltonian)
        self.assertEqual(og_op, lr_operators, lr_ops_og)
