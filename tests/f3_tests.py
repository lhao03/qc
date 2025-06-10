import unittest

import numpy as np
from openfermion import (
    count_qubits,
)

from min_part.f_3_ops import get_one_body_parts
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class FluidFragmentTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.80
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op

    def test_get_one_body_parts(self):
        n = 5
        m = (n * (n + 1)) // 2
        fake_h2 = np.random.rand(m)
        diags = [fake_h2[0], fake_h2[5], fake_h2[9], fake_h2[12], fake_h2[14]]
        self.assertEqual(diags, get_one_body_parts(np.array(fake_h2)))

    def test_convert_one_body_to_f3(self):
        pass

    def test_convert_two_body_to_f3(self):
        pass

    def test_move_from_2b_2_1b(self):
        pass

    def test_move_from_2b_2_1b_multiple(self):
        pass

    def test_rediag_1b(self):
        pass
