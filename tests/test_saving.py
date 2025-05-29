import unittest

from openfermion import (
    count_qubits,
)

from min_part.ham_decomp import gfro_decomp
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op
from min_part.utils import open_frags, save_frags


class SavingTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.8
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op

    def test_save_and_load(self):
        gfro_frags = gfro_decomp(tbt=self.H_tbt)
        save_frags(gfro_frags, "test")
        loaded_gfro_frags = open_frags("test")
        self.assertEqual(
            [f.operators for f in gfro_frags], [f.operators for f in loaded_gfro_frags]
        )
