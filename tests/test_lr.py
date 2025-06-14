import random
import unittest
from functools import reduce

import numpy as np
import torch
from hypothesis import given, settings
from openfermion import (
    count_qubits,
)
from min_part.gfro_decomp import make_fr_tensor_from_u
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.julia_ops import (
    rowwise_reshape,
    vecs2mat_reshape,
    reshape_eigs,
    eigen_jl,
    check_lr_decomp,
)
from min_part.lr_decomp import (
    make_supermatrix,
    four_tensor_to_two_tensor_indices,
    lr_decomp,
)
from min_part.molecules import mol_h2
from min_part.tensor import get_n_body_tensor, symmetricND, artifical_h2_tbt
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op
from min_part.utils import do_lr_fo

torch.set_default_tensor_type(torch.DoubleTensor)


settings.register_profile("fast", deadline=None)

# any tests executed before loading this profile will still use the
# default active profile of 100 examples.

settings.load_profile("fast")

# any tests executed after this point will use the active fast
# profile of 10 examples.


class DecompTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.80
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
        a = np.array(
            [
                [
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                ],
                [
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                ],
                [
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                ],
                [
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                ],
            ]
        )
        vals, vecs = eigen_jl(a)
        cur_Ds, cur_Ls = np.linalg.eig(a)
        idx = vals.argsort()[::-1]
        vals = vals[idx]
        sorted_vecs = vecs[:, idx]
        idx = cur_Ds.argsort()[::-1]
        cur_Ds = cur_Ds[idx]
        sort_cur_Ls = cur_Ls[:, idx]
        np.testing.assert_(cur_Ds, vals)
        np.testing.assert_array_almost_equal(
            sort_cur_Ls @ np.diagflat(cur_Ds) @ np.linalg.inv(sort_cur_Ls),
            sorted_vecs @ np.diagflat(vals) @ np.linalg.inv(sorted_vecs),
        )
        np.testing.assert_array_almost_equal(sort_cur_Ls, sorted_vecs)

    def test_inner_and_eig_reshape(self):
        n = 4
        sup_mat = self.H_tbt.reshape((n**2, n**2))
        cur_Ds, cur_Ls = np.linalg.eig(sup_mat)
        jl_D, jl_L = eigen_jl(sup_mat)
        print(jl_L[:, 0])
        Ls = [jl_L[:, i].reshape((n, n)) for i in range(len(jl_L))]
        Ls_jl = vecs2mat_reshape(jl_L, n)
        for i in range(n * n):
            py_l, jl_l = Ls[i], Ls_jl[i]
            print(py_l)
            np.testing.assert_equal(
                py_l,
                jl_l,
            )
            d_jl, U_jl = eigen_jl(jl_l)
            d_py, U_py = eigen_jl(py_l)
            py_d, jl_d = d_py.reshape((len(d_py), 1)), reshape_eigs(d_jl)
            np.testing.assert_allclose(
                py_d,
                jl_d,
            )

    @given(
        artifical_h2_tbt().filter(lambda m: not np.allclose(m, np.zeros((4, 4, 4, 4))))
    )
    def test_lr_decomp_h2_zeros(self, fake_h2):
        tol = 1e-7
        sym_ten = fake_h2
        # a = np.array(
        #     [
        #         [
        #             [
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #             ],
        #             [
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #             ],
        #         ],
        #         [
        #             [
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #             ],
        #             [
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #                 [0.0 + 0.0j, 0.0 + 0.0j],
        #             ],
        #         ],
        #     ]
        # )
        # a[0, 1, 0, 0] = 1
        # a[0, 1, 1, 0] = 1
        # a[1, 1, 1, 0] = 1
        # a[1, 1, 0, 1] = 1
        # print(a)
        # sym_ten = a
        # print(a.shape)
        np.testing.assert_allclose(sym_ten, sym_ten.T, atol=tol)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params])
        np.testing.assert_allclose(py_ten, sym_ten, atol=tol, rtol=tol)

        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [get_n_body_tensor(l.operators, 2, 4) for l in lr_frags_details_jl],
            # torch.zeros(*[size] * 4),
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
        np.testing.assert_allclose(sym_ten, sym_ten.T, atol=tol)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params])
        np.testing.assert_allclose(py_ten, sym_ten, atol=tol, rtol=tol)

        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [get_n_body_tensor(l.operators, 2, 4) for l in lr_frags_details_jl],
            # torch.zeros(*[size] * 4),
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
                get_n_body_tensor(lr_frag.operators, 2, 4), pt_t
            )
        lr_operators = reduce(
            lambda a, b: a + b, [f.operators for f in lr_frags_details]
        )
        og_op = tbt2op(fake_hamiltonian)
        self.assertEqual(og_op, lr_operators, lr_ops_og)

    def test_broken_case(self):
        sym_ten = self.H_tbt
        np.testing.assert_equal(sym_ten, sym_ten.T)
        lr_fo, lr_params = do_lr_fo(sym_ten)
        py_ten = reduce(lambda a, b: a + b, [p[2] for p in lr_params])
        np.testing.assert_array_almost_equal(py_ten, sym_ten)
        check_lr_decomp(sym_ten, [p[2] for p in lr_params])
        lr_frags_details_jl = lr_decomp(sym_ten)
        jl_ten = reduce(
            lambda a, b: a + b,
            [get_n_body_tensor(l.operators, 2, 4) for l in lr_frags_details_jl],
        )
        np.testing.assert_array_almost_equal(jl_ten, sym_ten)
