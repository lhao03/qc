import unittest

import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    FermionOperator,
    s_squared_operator,
)

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


class OperatorTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.8
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        H_ob_op = obt2op(H_obt)
        H_tb_op = tbt2op(H_tbt)
        H_ele = H_const + H_ob_op + H_tb_op
        self.eigenvalues, self.eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(H_ele)).toarray()
        )
        self.s_2_of = qubit_operator_sparse(
            jordan_wigner(
                s_squared_operator(n_spatial_orbitals=(self.n_qubits + 1) // 2)
            )
        )

    # === Particle Number ===
    def test_fermion_occ_num_op(self):
        n = get_particle_number(self.eigenvectors[:, 0], 4)
        self.assertEqual(2, n)

    # === Projected Spin ===
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

    # === Total Spin ===
    def test_total_spin_operator_construction(self):
        s_2 = make_total_spin_operator(2)
        self.assertEqual(
            s_squared_operator(n_spatial_orbitals=(self.n_qubits + 1) // 2),
            s_2
        )

    def test_total_spin_operator_1_elec(self):
        s = get_total_spin(self.eigenvectors[:, 4], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 4]), s)

    def test_total_spin_operator_2_elec(self):
        s = get_total_spin(self.eigenvectors[:, 0], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 0]), s)

    def test_total_spin_operator_3_elec(self):
        s = get_total_spin(self.eigenvectors[:, 6], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 6]), s)

    def test_total_spin_operator_4_elec(self):
        s = get_total_spin(self.eigenvectors[:, -1], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, -1]), s)

    # === Operator Utils ===
    def test_square_operator(self):
        a_creat = FermionOperator("0^", 3.1)
        a_annih = FermionOperator("3", 1.2)
        fo_add = a_creat + a_annih
        fo_min = a_annih - a_creat
        self.assertEqual(
            FermionOperator("0^ 0^", 3.1**2)
            + FermionOperator("0^ 3", 1.2 * 3.1)
            + FermionOperator("3 0^", 1.2 * 3.1)
            + FermionOperator("3 3", 1.2**2),
            get_squared_operator(fo_add),
        )
        self.assertEqual(
            FermionOperator("3 3", 1.2**2)
            - FermionOperator("3 0^", 1.2 * 3.1)
            - FermionOperator("0^ 3", 1.2 * 3.1)
            + FermionOperator("0^ 0^", 3.1**2),
            get_squared_operator(fo_min),
        )
