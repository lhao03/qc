import unittest

import scipy as sp
from openfermion import count_qubits, jordan_wigner, qubit_operator_sparse

from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.operators import get_particle_number, get_projected_spin, get_total_spin
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class OperatorTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.8
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        n_qubits = count_qubits(H)
        H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
        H_ob_op = obt2op(H_obt)
        H_tb_op = tbt2op(H_tbt)
        H_ele = H_const + H_ob_op + H_tb_op
        self.eigenvalues, self.eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(H_ele)).toarray()
        )

    def test_fermion_occ_num_op(self):
        n = get_particle_number(self.eigenvectors[:, 0], 4)
        self.assertEqual(2, n)

    def test_projected_spin_operator_singlet_state(self):
        s = get_projected_spin(self.eigenvectors[:, 0], 2)
        self.assertEqual(0, s)

    def test_projected_spin_operator_1_elec(self):
        s = get_projected_spin(self.eigenvectors[:, 4], 2)
        self.assertEqual(0.5, s)

    def test_projected_spin_operator_3_elec(self):
        s = get_projected_spin(self.eigenvectors[:, 6], 2)
        self.assertEqual(0.5, s)

    def test_projected_spin_operator_4_elec(self):
        s = get_projected_spin(self.eigenvectors[:, -1], 2)
        self.assertEqual(0, s)

    def test_total_spin_operator_1_elec(self):
        s = get_total_spin(self.eigenvectors[:, 4], 2)
        self.assertEqual(0, s)

    def test_total_spin_operator_2_elec(self):
        s = get_total_spin(self.eigenvectors[:, 0], 2)
        self.assertEqual(0, s)

    def test_total_spin_operator_3_elec(self):
        s = get_total_spin(self.eigenvectors[:, 6], 2)
        self.assertEqual(0, s)

    def test_total_spin_operator_4_elec(self):
        s = get_total_spin(self.eigenvectors[:, -1], 2)
        self.assertEqual(0, s)
