import unittest

import numpy as np
from openfermion import count_qubits, FermionOperator

from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor import get_n_body_tensor, get_n_body_fo, is_chemist_ordered
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class ConversionTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.8
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op

    # == utils ++
    def test_ordering(self):
        self.assertTrue(is_chemist_ordered(((1, 1), (0, 0), (1, 1), (0, 0))))
        self.assertFalse(is_chemist_ordered(((0, 0), (1, 1), (0, 0))))
        self.assertFalse(is_chemist_ordered(((1, 1), (0, 0), (1, 1),)))
        self.assertFalse(is_chemist_ordered(((1, 0), (0, 1), (1, 0), (0, 1))))

   # === `FermionOperator` to Tensor tests
    def test_fake_1b_fo_2_tensor(self):
        coeff = -1.7
        fake_2b_fo = FermionOperator('3^ 1 2^ 0', coeff)
        tensor = get_n_body_tensor(fake_2b_fo, 2, 4)
        fo = tbt2op(tensor)
        self.assertEqual(tensor[3][1][2][0], coeff)
        self.assertEqual(fake_2b_fo, fo)

    def test_1b_fo_2_tensor(self):
        tensor = get_n_body_tensor(self.H_ob_op, 1, 4)
        self.assertTrue(np.array_equal(tensor, self.H_obt))

    def test_2b_fo_2_tensor(self):
        tensor = get_n_body_tensor(self.H_tb_op, 2, 4)
        self.assertTrue(np.array_equal(self.H_tbt, tensor))

    def test_3b_fo_2_tensor(self):
        pass

    # === Tensor to `FermionOperator` tests
    def test_tensor_2_1b_fo(self):
        fo = get_n_body_fo(self.H_obt)
        self.assertEqual(fo, self.H_ob_op)

    def test_tensor_2_2b_fo(self):
        fo = get_n_body_fo(self.H_tbt)
        self.assertEqual(fo, self.H_tb_op)

    def test_tensor_2_3b_fo(self):
        pass

