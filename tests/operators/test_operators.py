import unittest

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    FermionOperator,
    s_squared_operator,
    expectation,
    number_operator,
    sz_operator,
    eigenspectrum,
)

from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2, mol_h4, mol_n2
from min_part.operators import (
    get_particle_number,
    get_projected_spin,
    get_total_spin,
    get_squared_operator,
    extract_eigenvalue,
    make_total_spin_operator,
    collapse_to_number_operator,
    subspace_restriction,
)
from min_part.tensor import obt2op, tbt2op
from tests.utils.sim_tensor import get_chem_tensors


class OperatorTest(unittest.TestCase):
    def setUp(self):
        bond_length = 1
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        H_ob_op = obt2op(H_obt)
        H_tb_op = tbt2op(H_tbt)
        self.H_ele = H_const + H_ob_op + H_tb_op
        self.eigenvalues, self.eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(self.H_ele)).toarray()
        )
        self.s_2_of = qubit_operator_sparse(
            jordan_wigner(
                s_squared_operator(n_spatial_orbitals=(self.n_qubits + 1) // 2)
            )
        )
        # h4
        self.h4_mol = mol_h4(bond_length)
        H_4, num_elecs_h4 = obtain_OF_hamiltonian(self.h4_mol)
        self.h4_qubits = count_qubits(H_4)
        H4_const, H4_obt, H4_tbt = get_chem_tensors(H_4, N=self.h4_qubits)
        self.H_4ele = H4_const + obt2op(H4_obt) + tbt2op(H_tbt)
        # n2
        self.n2_mol = mol_n2(bond_length)
        H_N2, num_elecs_n2 = obtain_OF_hamiltonian(self.h4_mol)
        self.n2_qubits = count_qubits(H_N2)
        N_const, N_obt, N_tbt = get_chem_tensors(H_N2, N=self.n2_qubits)
        self.H_N2_ele = N_const + obt2op(N_obt) + tbt2op(N_tbt)

    # === Particle Number ===
    def test_fermion_occ_num_op(self):
        n = get_particle_number(self.eigenvectors[:, 0], 4)
        self.assertEqual(2, n)
        no = qubit_operator_sparse(jordan_wigner(number_operator(n_modes=4)))
        of_expectation_value = expectation(operator=no, state=self.eigenvectors[:, 0])
        self.assertEqual(2, round(of_expectation_value))

    # === Projected Spin ===
    def test_projected_spin_operator_singlet_state(self):
        s = get_projected_spin(self.eigenvectors[:, 0], 2)
        sz = qubit_operator_sparse(jordan_wigner(sz_operator(2)))
        self.assertEqual(0, s)
        self.assertEqual(0, round(expectation(sz, self.eigenvectors[:, 0])))

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
            s_squared_operator(n_spatial_orbitals=(self.n_qubits + 1) // 2), s_2
        )

    def test_total_spin_operator_1_elec(self):
        s = get_total_spin(self.eigenvectors[:, 4], 2)
        print(eigenspectrum(sz_operator(2)))
        s2 = qubit_operator_sparse(jordan_wigner(s_squared_operator(2)))
        self.assertEqual(
            extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 4], panic=True), s
        )
        self.assertEqual(
            s, round(expectation(operator=s2, state=self.eigenvectors[:, 4]))
        )

    def test_total_spin_operator_2_elec(self):
        s = get_total_spin(self.eigenvectors[:, 0], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 0]), s)

    def test_total_spin_operator_3_elec(self):
        s = get_total_spin(self.eigenvectors[:, 6], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, 6]), s)

    def test_total_spin_operator_4_elec(self):
        s = get_total_spin(self.eigenvectors[:, -1], 2)
        self.assertEqual(extract_eigenvalue(self.s_2_of, self.eigenvectors[:, -1]), s)

    def test(self):
        for i in range(16):
            print(self.eigenvalues[i], get_total_spin(self.eigenvectors[:, i], 2))

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

    # == Number Operator ===
    def test_num_op_collapose(self):
        self.assertEqual(
            FermionOperator("1^ 1 2^ 2")
            + FermionOperator("3^ 2 3^ 2")
            + FermionOperator("0^ 0"),
            collapse_to_number_operator(
                FermionOperator("1^ 1 2^ 2")
                + FermionOperator("3^ 2 3^ 2")
                + FermionOperator("0^ 0 0^ 0")
            ),
        )

        self.assertEqual(
            FermionOperator("1^ 1 3^ 2 3^ 2")
            + FermionOperator("3^ 2 1^ 1 3^ 2")
            + FermionOperator("3^ 2 1^ 1 3^ 2 0^ 0 3^ 2 3^ 2"),
            collapse_to_number_operator(
                FermionOperator("1^ 1 1^ 1 3^ 2 3^ 2")
                + FermionOperator("3^ 2 1^ 1 3^ 2")
                + FermionOperator("3^ 2 1^ 1 3^ 2 0^ 0 0^ 0 3^ 2 3^ 2"),
            ),
        )

        self.assertEqual(
            FermionOperator("1^ 1 3^ 2 3^ 2")
            + FermionOperator("3^ 2 1^ 1 3^ 2")
            + FermionOperator("1^ 1 0^ 0"),
            collapse_to_number_operator(
                FermionOperator("1^ 1 1^ 1 3^ 2 3^ 2")
                + FermionOperator("3^ 2 1^ 1 3^ 2")
                + FermionOperator("1^ 1 1^ 1 0^ 0 0^ 0"),
            ),
        )

        self.assertEqual(
            FermionOperator("1^ 1")
            + FermionOperator("3^ 2 1^ 1 3^ 2")
            + FermionOperator("1^ 1 0^ 0"),
            collapse_to_number_operator(
                FermionOperator("1^ 1")
                + FermionOperator("3^ 2 1^ 1 3^ 2")
                + FermionOperator("1^ 1 1^ 1 0^ 0 0^ 0"),
            ),
        )

    # === Projection Operator ===
    def test_projection_operator(self):
        ss_H_0 = subspace_restriction(
            self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=0
        )
        ss_H_1 = subspace_restriction(
            self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=1
        )
        ss_H_2 = subspace_restriction(
            self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=2
        )
        ss_H_3 = subspace_restriction(
            self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=3
        )
        ss_H_4 = subspace_restriction(
            self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=4
        )
        for i in range(len(self.eigenvalues)):
            print(f"energy: {self.eigenvalues[i]}")
            print(f"num elecs: {get_particle_number(self.eigenvectors[:, i], 4)}")
            print(f"s2: {get_total_spin(self.eigenvectors[:, i], 2)}")
            print(f"sz: {get_projected_spin(self.eigenvectors[:, i], 2)}")
        ss_opers = [ss_H_0, ss_H_1, ss_H_2, ss_H_3, ss_H_4]
        for i, ss in enumerate(ss_opers):
            vals, vecs = np.linalg.eigh(ss.toarray())
            print(f"energies for {i} elecs, sz=0, s2=0: {vals}")

    def test_cicd_projection_operator(self):
        # == exact for h2
        for i in range(self.n_qubits + 1):
            print("exact for h2")
            ss = subspace_restriction(
                self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=i
            )
            vals, vecs = np.linalg.eigh(ss.toarray())
            print(f"energies for {i} elecs, sz=0, s2=0: {vals}")
            print("cisd for h2")
            ss = subspace_restriction(
                self.H_ele, n_spin_orbs=self.n_qubits, num_elecs=i, ci_projection=2
            )
            vals, vecs = np.linalg.eigh(ss.toarray())
            print(f"energies for {i} elecs, sz=0, s2=0: {vals}")

        # ==  n2
        gs_n2 = min(
            np.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(self.H_N2_ele)).toarray()
            )[0]
        )
        print(f"ground state of n2: {gs_n2}")
        for i in range(self.n2_qubits + 1):
            print("exact for n2")
            ss = subspace_restriction(
                self.H_N2_ele,
                n_spin_orbs=self.n2_qubits,
                num_elecs=i,
            )
            vals, vecs = np.linalg.eigh(ss.toarray())
            print(f"energies for {i} elecs, sz=0, s2=0: {vals}")
            print("cisd for n2")
            ss = subspace_restriction(
                self.H_N2_ele, n_spin_orbs=self.n2_qubits, num_elecs=i, ci_projection=2
            )
            vals, vecs = np.linalg.eigh(ss.toarray())
            print(f"energies for {i} elecs, sz=0, s2=0: {vals}")
