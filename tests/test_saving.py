import os
import unittest
from typing import List

from openfermion import (
    count_qubits,
)

from d_types.fragment_types import GFROFragment
from min_part.gfro_decomp import gfro_decomp
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2, h2_settings
from min_part.tensor import obt2op, tbt2op, get_chem_tensors
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
        loaded_gfro_frags = open_frags("test.pkl")
        self.assertEqual(
            [f.operators for f in gfro_frags], [f.operators for f in loaded_gfro_frags]
        )

    def test_load_files(self):
        global gfro_files

        config_settings = h2_settings
        global_id = "42"
        parent_dir = f"../data/{config_settings.mol_name.lower()}"
        child_dir = os.path.join(parent_dir, "06-09", str(global_id))

        load = True
        if load:
            gfro_files, lr_files = get_saved_file_names(child_dir)

        for bond_length, gfro_file in zip(config_settings.xpoints, gfro_files):
            print(f"thetas and lambdas for {bond_length}")
            gfro_data: List[GFROFragment] = open_frags(gfro_file)
            self.assertTrue(isinstance(gfro_data[0], GFROFragment))
            for g_d in gfro_data:
                print(f"thetas: {g_d.thetas}")
                print(f"lambdas: {g_d.lambdas}")
