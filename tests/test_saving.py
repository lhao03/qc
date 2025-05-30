import unittest

from openfermion import (
    count_qubits,
)

from min_part.ham_decomp import gfro_decomp
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op
from min_part.typing import GFROFragment
from min_part.utils import open_frags, save_frags, get_saved_file_names


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

    def test_load_files(self):
        saved_parent_folder = "../data/h4/05-29/11"
        gfro_name = "gfro_<built-in function id>_H4"
        lr_name = "lr_<built-in function id>_H4"
        gfro_files, lr_files = get_saved_file_names(
            saved_parent_folder, gfro_name, lr_name
        )
        for gfro in gfro_files:
            gfro_frags = open_frags(gfro)
            self.assertTrue(isinstance(gfro_frags[0], GFROFragment))
