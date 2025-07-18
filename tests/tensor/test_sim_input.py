import unittest

import numpy as np
from hypothesis import given
from openfermion import count_qubits
from opt_einsum import contract

from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_n2
from tests.utils.sim_tensor import (
    generate_symm_unitary_matrices,
    make_tensors_h2,
    get_chem_tensors,
)


def make_tensors_n2(bond_length):
    mol = mol_n2(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


class TensorTest(unittest.TestCase):
    @given(generate_symm_unitary_matrices(n=4))
    def test_uni_decomp(self, u_symm):
        diags = u_symm[0]
        u = u_symm[1]
        symm = u_symm[2]
        np.testing.assert_array_almost_equal(
            u @ np.diagflat(diags) @ u.T, u.T @ np.diagflat(diags) @ u
        )
        np.testing.assert_array_almost_equal(
            contract("r,rp,rq->pq", diags, u, u),
            contract("r,pr,qr->pq", diags, u, u),
        )
        # ===
        vals, vecs = np.linalg.eigh(symm)
        np.testing.assert_array_almost_equal(
            contract("r,pr,qr->pq", vals, vecs, vecs),
            contract("r,qr,pr->pq", vals, vecs, vecs),
        )
        try:
            np.testing.assert_array_almost_equal(vecs, vecs.T)
            np.testing.assert_array_almost_equal(
                vecs @ np.diagflat(vals) @ vecs.T, vecs.T @ np.diagflat(vals) @ vecs
            )
            np.testing.assert_array_almost_equal(vecs, np.linalg.inv(vecs))
            np.testing.assert_array_almost_equal(
                contract("r,rp,rq->pq", vals, vecs, vecs),
                contract("r,pr,qr->pq", vals, vecs, vecs),
            )
            self.assertAlmostEqual(np.linalg.det(vecs), 1)
        except AssertionError:
            pass
        # ==

        np.testing.assert_array_almost_equal(
            contract("r,pr,qr->pq", diags, u, u),
            contract("r,pr,qr->pq", vals, vecs, vecs),
        )

    def test_molecules(self):
        H_const, H_obt, H_tbt = make_tensors_n2(1)
        H_const, H_obt, H_tbt = make_tensors_h2(1)
