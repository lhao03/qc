import unittest

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    FermionOperator,
    s_squared_operator,
)

from min_part.ham_decomp import make_supermatrix, four_tensor_to_two_tensor_indices
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.operators import (
    get_particle_number,
    get_projected_spin,
    get_total_spin,
    get_squared_operator,
    extract_eigenvalue,
    make_total_spin_operator,
)
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


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

    # === Helpers ===
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
