import os
import pickle
import unittest
from pathlib import Path
from typing import Union

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    s_squared_operator,
    get_number_preserving_sparse_operator,
    QubitOperator,
    FermionOperator,
    normal_ordered,
    reverse_jordan_wigner,
    get_sparse_operator,
    get_ground_state,
)

from pert_trotter import config
from pert_trotter.energy_frag_utils import sum_frags
from pert_trotter.fermi_frag import Do_Fermi_Partitioning
from pert_trotter.ham_utils import obtain_OF_hamiltonian
from pert_trotter.tensor_utils import get_chem_tensors, obt2op, tbt2op

methods = ["lr", "gfro", "lrlcu", "gfrolcu", "sdgfro"]


def save_state(mol_name: str, eigenvalues, eigenvectors):
    jw_of = os.path.join(os.getcwd(), "data", "JW_OF", mol_name)
    if not os.path.exists(jw_of):
        os.mkdir(jw_of)
    p1 = os.path.join(jw_of, f"{mol_name}_v")
    p2 = os.path.join(jw_of, f"{mol_name}_w")
    with open(p1, "wb") as out_file:
        pickle.dump(eigenvalues, out_file)
    with open(p2, "wb") as out_file:
        pickle.dump(eigenvectors, out_file)


def get_projected_sparse_op_non_tapered(
    H_OF: Union[FermionOperator, QubitOperator],
    n_qubits,
    num_elecs,
    nsz2ssq_proj_sparse,
    spin_preserving=True,
    excitation_level=None,
    reference_determinant=None,
):
    # H_OF is a FermionOperator in full space
    if isinstance(H_OF, QubitOperator):
        H_OF = normal_ordered(reverse_jordan_wigner(H_OF))
    first_projected_op = get_number_preserving_sparse_operator(
        H_OF,
        n_qubits,
        num_elecs,
        spin_preserving=spin_preserving,
        excitation_level=excitation_level,
        reference_determinant=reference_determinant,
    )
    return nsz2ssq_proj_sparse * first_projected_op * nsz2ssq_proj_sparse.T


class FragmentTest(unittest.TestCase):
    def template(self, mol_name, mol, energy, elecs, qubits):
        # make the molecular hamiltonian
        H, num_elecs = obtain_OF_hamiltonian(mol)
        self.assertEqual(elecs, num_elecs)
        n_qubits = count_qubits(H)  # 1s 1s
        self.assertEqual(qubits, n_qubits)  # one per each spin orbital
        H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
        H_ob_op = obt2op(H_obt)
        H_tb_op = tbt2op(H_tbt)
        H_ele = H_const + H_ob_op + H_tb_op
        jw_op = jordan_wigner(H_ele)
        gs_energy = get_ground_state(get_sparse_operator(jw_op))[0]
        self.assertAlmostEqual(gs_energy, energy, places=2)
        config.mol_name = mol_name
        config.n_qubits = n_qubits
        config.num_elecs = num_elecs
        # generate the exact eigenstates
        jw_op_array_sparse = qubit_operator_sparse(jw_op)
        jw_op_array = jw_op_array_sparse.toarray()
        eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
        eigenvalue_0 = eigenvalues[0]
        self.assertAlmostEqual(eigenvalue_0, gs_energy)
        eigenvectors_0 = eigenvectors[:, [0]]
        eigenvectors_0_sparse = sp.sparse.csc_matrix(eigenvectors_0)
        save_state(mol_name, eigenvalues, eigenvectors)
        # build projector
        s_sq = s_squared_operator(n_qubits // 2)
        s_sq_sparse = get_number_preserving_sparse_operator(
            s_sq, n_qubits, num_elecs, spin_preserving=True
        )
        s_sq_array = s_sq_sparse.toarray()
        s_sq_values, s_sq_vectors = np.linalg.eigh(s_sq_array)
        s_sq_vectors_sparse = sp.sparse.csc_matrix(s_sq_vectors)
        non_cisd_dim = len(list(filter(lambda n: n <= 0.01, s_sq_values)))
        s_sq_evals, nsz2ssq_proj = (
            s_sq_values[:non_cisd_dim],
            s_sq_vectors[:, :non_cisd_dim].T,
        )
        nsz2ssq_proj_sparse = sp.sparse.csc_matrix(nsz2ssq_proj)
        # hamiltonian projection, lr
        method = methods[0]
        method_path = f"data/{method}/sparse_arrays/full_space/"
        full_path = os.path.join(os.getcwd(), method_path)
        Path(full_path).mkdir(parents=True, exist_ok=True)
        spatial = True
        save = True
        load = False
        fragment_array_sarray = Do_Fermi_Partitioning(
            H_ele,
            type=method,
            tol=1e-6,
            spacial=spatial,
            save=True,
            load=load,
            projector_func=lambda f,
            excitation_level: get_projected_sparse_op_non_tapered(
                H_OF=f,
                nsz2ssq_proj_sparse=nsz2ssq_proj_sparse,
                n_qubits=n_qubits,
                num_elecs=num_elecs,
                excitation_level=excitation_level,
            ),
        )
        summed_energies = sum_frags(fragment_array_sarray)
        print(summed_energies)
        self.assertTrue(gs_energy >= summed_energies)

    def test_h2(self):
        self.template(
            mol_name="h2",
            mol=[["H", [0, 0, 0]], ["H", [0, 0, 1]]],
            energy=-1.10,
            elecs=2,
            qubits=4,
        )

    def test_lih(self):
        self.template(
            mol_name="lih",
            mol=[["Li", [0, 0, 0]], ["H", [0, 0, 1.5949]]],
            energy=-7.88,
            elecs=4,
            qubits=12,
        )

    def test_n2(self):
        self.template(
            mol_name="n2",
            mol=[["N", [0, 0, 0]], ["N", [0, 0, 1.0977]]],
            energy=-7.88,
            elecs=14,  # N: 1s 2s 2p
            qubits=20,
        )

    def test_h4(self):
        self.template(
            mol_name="h4",
            mol= [
            ["H", [0, 0, 0]],
            ["H", [0, 0, 1]],
            ["H", [0, 0, 2]],
            ["H", [0, 0, 3]],
        ],
            energy=-2.166,
            elecs=4,
            qubits=8,
        )
