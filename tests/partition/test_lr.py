import random
import unittest
from functools import reduce

import numpy as np
import scipy as sp
import torch
from hypothesis import given, settings
from numpy import isclose
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    s_squared_operator,
    sz_operator,
)

from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.julia_ops import (
    rowwise_reshape,
    vecs2mat_reshape,
    reshape_eigs,
    eigen_jl,
    check_lr_decomp,
)

from min_part.molecules import mol_h2
from min_part.operators import (
    get_projected_spin,
    get_total_spin,
    extract_eigenvalue,
)
from min_part.tensor import (
    get_n_body_tensor_chemist_ordering,
    obt2op,
    tbt2op,
    make_x_matrix,
    extract_thetas,
    make_unitary_im,
    make_fr_tensor_from_u,
    make_lambda_matrix,
)

from min_part.utils import do_lr_fo
from tests.utils.sim_tensor import (
    get_chem_tensors,
    symmetricND,
    artifical_h2_tbt,
)

torch.set_default_tensor_type(torch.DoubleTensor)
settings.register_profile("fast", deadline=None)
settings.load_profile("fast")


class DecompTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.75
        self.mol = mol_h2(bond_length)
        self.H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(self.H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(
            H=self.H, N=self.n_qubits
        )
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op

    # === Low Rank Helpers ===
    def test_4_to_2_indices(self):
        pq, rs = four_tensor_to_two_tensor_indices(0, 0, 1, 4, n=5)
        self.assertEqual(pq, 0)
        self.assertEqual(rs, 9)

    def test_supermatrix_2(self):
        test_matrix = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
            ]
        )
        supermatrix = make_supermatrix(test_matrix)
        self.assertEqual(test_matrix[0][0][1][1], supermatrix[0][3])

    def test_supermatrix_4(self):
        n = 13
        test_matrix = np.random.rand(n, n, n, n)
        supermatrix = make_supermatrix(test_matrix)
        jl_supermatrix = rowwise_reshape(test_matrix, n * n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        pq, rs = four_tensor_to_two_tensor_indices(i, j, k, l, n)
                        self.assertEqual(test_matrix[i, j, k, l], supermatrix[pq][rs])
                        self.assertEqual(
                            test_matrix[i, j, k, l], jl_supermatrix[pq][rs]
                        )

    def test_eign(self):
        n = 4
        jl_flat = rowwise_reshape(self.H_tbt, n * n)
        sup_mat = self.H_tbt.reshape((n * n, n * n))
        np.testing.assert_equal(jl_flat, sup_mat)
        vals, vecs = eigen_jl(jl_flat)
        cur_Ds, cur_Ls = np.linalg.eig(sup_mat)
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        sorted_vecs = vecs[:, idx]
        idx = cur_Ds.argsort()[::-1]
        cur_Ds = cur_Ds[idx]
        sort_cur_Ls = cur_Ls[:, idx]
        np.testing.assert_array_almost_equal(cur_Ds, vals)
        np.testing.assert_array_almost_equal(
            sort_cur_Ls @ np.diagflat(cur_Ds) @ np.linalg.inv(sort_cur_Ls),
            sorted_vecs @ np.diagflat(vals) @ np.linalg.inv(sorted_vecs),
        )

    def test_inner_and_eig_reshape(self):
        n = 4
        sup_mat = self.H_tbt.reshape((n**2, n**2))
        cur_Ds, cur_Ls = np.linalg.eig(sup_mat)
        jl_D, jl_L = eigen_jl(sup_mat)
        Ls = [jl_L[:, i].reshape((n, n)) for i in range(len(jl_L))]
        Ls_jl = vecs2mat_reshape(jl_L, n)
        for i in range(n * n):
            py_l, jl_l = Ls[i], Ls_jl[i]
            np.testing.assert_equal(
                py_l,
                jl_l,
            )
            d_jl, U_jl = eigen_jl(jl_l)
            d_py, U_py = eigen_jl(py_l)
            py_d, jl_d = d_py.reshape((len(d_py), 1)), reshape_eigs(d_jl)
            np.testing.assert_array_almost_equal(
                py_d,
                jl_d,
            )

    @given(
        artifical_h2_tbt().filter(lambda m: not np.allclose(m, np.zeros((4, 4, 4, 4))))
    )
    def test_lr_decomp_h2_zeros(self, fake_h2):
        tol = 1e-7
        sym_ten = fake_h2
        empty = np.zeros((4, 4, 4, 4))
        np.testing.assert_array_equal(sym_ten, sym_ten.T)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params], empty)
        np.testing.assert_allclose(py_ten, sym_ten, atol=tol, rtol=tol)

        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [
                get_n_body_tensor_chemist_ordering(l.operators, 2, 4)
                for l in lr_frags_details_jl
            ],
            empty,
        )
        np.testing.assert_allclose(jl_ten, sym_ten, atol=tol, rtol=tol)
        np.testing.assert_allclose(jl_ten, py_ten, atol=tol, rtol=tol)

    @given(
        symmetricND(size=4).filter(
            lambda t: not torch.allclose(t, torch.zeros(*[4] * 4))
        ),
    )
    def test_lr_decomp_h2(self, sym_ten):
        tol = 1e-7
        sym_ten = sym_ten.detach().cpu().numpy()
        empty = np.zeros((4, 4, 4, 4))
        np.testing.assert_allclose(sym_ten, sym_ten.T, atol=tol)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params], empty)
        np.testing.assert_allclose(py_ten, sym_ten, atol=tol, rtol=tol)

        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [
                get_n_body_tensor_chemist_ordering(l.operators, 2, 4)
                for l in lr_frags_details_jl
            ],
            empty,
        )
        np.testing.assert_allclose(jl_ten, sym_ten, atol=tol, rtol=tol)
        np.testing.assert_allclose(jl_ten, py_ten, atol=tol, rtol=tol)

    def test_lr_decomp_fake(self):
        n = 4
        m = n * (n + 1) // 2
        fake_u = np.array(
            [
                [0.70710029, 0.00303002, 0.70710028, 0.00303002],
                [-0.00303002, 0.70710029, -0.00303002, 0.70710028],
                [-0.70710493, -0.00161596, 0.70710494, 0.00161596],
                [0.00161597, -0.70710493, -0.00161596, 0.70710494],
            ]
        )
        fake_lambdas = np.array(sorted([0.1 * random.randint(1, 10) for _ in range(m)]))
        fake_hamiltonian = make_fr_tensor_from_u(fake_lambdas, fake_u, n)
        lr_frags_details = lr_decomp(fake_hamiltonian)
        lr_fo, lr_params = do_lr_fo(fake_hamiltonian)
        lr_ops_og = reduce(lambda a, b: a + b, lr_fo)
        for i, lr_frag in enumerate(lr_frags_details):
            pt_t = lr_params[i][2]
            np.testing.assert_array_almost_equal(
                get_n_body_tensor_chemist_ordering(lr_frag.operators, 2, 4), pt_t
            )
        lr_operators = reduce(
            lambda a, b: a + b, [f.operators for f in lr_frags_details]
        )
        og_op = tbt2op(fake_hamiltonian)
        self.assertEqual(og_op, lr_operators, lr_ops_og)

    def test_h2(self):
        sym_ten = self.H_tbt
        np.testing.assert_equal(sym_ten, sym_ten.T)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params])
        np.testing.assert_array_almost_equal(py_ten, sym_ten)
        check_lr_decomp(sym_ten, [p[2] for p in lr_params])
        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [
                get_n_body_tensor_chemist_ordering(l.operators, 2, 4)
                for l in lr_frags_details_jl
            ],
        )
        np.testing.assert_array_almost_equal(jl_ten, sym_ten)

    def test_lr_tensor_formation(self):
        lr_fo, lr_params = do_lr_fo(self.H_tbt)
        lr_frags_details_jl = lr_decomp(self.H_tbt)
        for fo, params in zip(lr_fo, lr_params):
            print("==")
            c, u, tensor = params
            thetas, diags = extract_thetas(u)
            x = make_x_matrix(thetas, 4)
            for i, d in enumerate(diags):
                if not isclose(d, 0):
                    x[i, i] = d
            u_exp = sp.linalg.logm(u)
            np.testing.assert_array_almost_equal(x, u_exp)
            made_u = make_unitary_im(thetas=thetas, diags=diags, n=4)
            np.testing.assert_array_almost_equal(made_u, u)
            np.testing.assert_array_almost_equal(
                get_lr_fragment_tensor_from_parts(c[0], c[1], thetas, diags), tensor
            )
        for l in lr_frags_details_jl:
            np.testing.assert_array_almost_equal(
                get_lr_fragment_tensor(l),
                get_n_body_tensor_chemist_ordering(l.operators, 2, 4),
            )

    def test_lr_h2_occs(self):
        f = lr_decomp(tbt=self.H_tbt)
        n = self.H_tbt.shape[0]
        for frag_details in f:
            diag_eigenvalues, diag_eigenvectors = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(frag_details.operators)).toarray()
            )
            occupations, eigenvalues = lr_fragment_occ(
                fragment=frag_details, num_spin_orbs=n
            )
            frag_details.to_fluid()
            c = np.reshape(frag_details.coeffs, (n, 1))
            c_matrix = frag_details.outer_coeff * c @ c.T
            np.testing.assert_array_almost_equal(
                make_lambda_matrix(frag_details.fluid_parts.static_lambdas, 4)
                + np.diag(frag_details.fluid_parts.fluid_lambdas),
                c_matrix,
            )
            occupations_0, eigenvalues_0 = get_expectation_vals_lr_frags(
                self=frag_details, num_spin_orbs=n, expected_e=0
            )
            occupations_1, eigenvalues_1 = get_expectation_vals_lr_frags(
                self=frag_details, num_spin_orbs=n, expected_e=1
            )
            occupations_2, eigenvalues_2 = get_expectation_vals_lr_frags(
                self=frag_details, num_spin_orbs=n, expected_e=2
            )
            occupations_3, eigenvalues_3 = get_expectation_vals_lr_frags(
                self=frag_details, num_spin_orbs=n, expected_e=3
            )
            occupations_4, eigenvalues_4 = get_expectation_vals_lr_frags(
                self=frag_details, num_spin_orbs=n, expected_e=4
            )
            eigs_f = (
                eigenvalues_0
                + eigenvalues_1
                + eigenvalues_2
                + eigenvalues_3
                + eigenvalues_4
            )
            for i in range(16):
                self.assertEqual(
                    get_total_spin(diag_eigenvectors[:, i], 2),
                    extract_eigenvalue(
                        qubit_operator_sparse(
                            jordan_wigner(s_squared_operator(n_spatial_orbitals=2))
                        ),
                        diag_eigenvectors[:, i],
                        panic=False,
                    ),
                )
                self.assertEqual(
                    get_projected_spin(diag_eigenvectors[:, i], 2),
                    extract_eigenvalue(
                        qubit_operator_sparse(
                            jordan_wigner(sz_operator(n_spatial_orbitals=2))
                        ),
                        diag_eigenvectors[:, i],
                        panic=False,
                    ),
                )

            np.testing.assert_array_almost_equal(np.sort(eigs_f), np.sort(eigenvalues))
            self.assertTrue(
                np.allclose(np.sort(diag_eigenvalues), np.sort(eigenvalues))
            )


from min_part.lr_decomp import (  # noqa: E402
    make_supermatrix,
    four_tensor_to_two_tensor_indices,
    lr_decomp,
    get_lr_fragment_tensor_from_parts,
    get_lr_fragment_tensor,
    lr_fragment_occ,
    get_expectation_vals_lr_frags,
)
