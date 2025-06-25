import unittest
from typing import List

import numpy as np
import scipy as sp
from hypothesis import given, example, settings
from opt_einsum import contract

from d_types.fragment_types import LRFragment, GFROFragment
from min_part.gfro_decomp import make_fr_tensor
from min_part.julia_ops import (
    jl_extract_thetas,
    jl_make_u_im,
    jl_make_x_im,
    jl_make_x,
    jl_make_u,
    jl_compare_matrices,
)
from min_part.lr_decomp import get_lr_fragment_tensor
from min_part.tensor import (
    extract_thetas,
    make_unitary_im,
    make_x_matrix,
    make_unitary,
    tbt2op,
)
from tests.testing_utils.sim_molecules import H_2_LR, H_2_GFRO
from tests.testing_utils.sim_tensor import (
    generate_symm_unitary_matrices,
)

settings.register_profile("slow", deadline=None, print_blob=True)
settings.load_profile("slow")


class UnitaryTest(unittest.TestCase):
    @example(
        (
            np.array([1.53193455, 1.625, 0.5, 1.0]),
            np.array(
                [
                    [1.00000000e00, -1.24311712e-16, -7.35416349e-17, -1.45443250e-16],
                    [-1.24311712e-16, 1.00000000e00, 2.16372734e-16, -1.71723751e-16],
                    [-7.35416349e-17, 2.16372734e-16, 1.00000000e00, 1.78106774e-16],
                    [-1.45443250e-16, -1.71723751e-16, 1.78106774e-16, 1.00000000e00],
                ]
            ),
            np.array(
                [
                    [1.53193455e00, -3.92443940e-16, -1.49431789e-16, -3.68252791e-16],
                    [-3.92443940e-16, 1.62500000e00, 4.59792060e-16, -4.50774848e-16],
                    [-1.49431789e-16, 4.59792060e-16, 5.00000000e-01, 2.67160161e-16],
                    [-3.68252791e-16, -4.50774848e-16, 2.67160161e-16, 1.00000000e00],
                ]
            ),
        ),
    )
    @given(generate_symm_unitary_matrices(n=4))
    def test_uni_decomp(self, u_symm):  # TODO: Ask for help next week
        diags, u, symm = u_symm
        np.testing.assert_array_almost_equal(
            contract("r,rp,rq->pq", diags, u, u),
            contract("r,pr,qr->pq", diags, u, u),
        )
        thetas, diag_theta = extract_thetas(u)
        made_u = make_unitary_im(thetas, diag_theta, 4)
        np.testing.assert_array_almost_equal(u, made_u)
        np.testing.assert_array_almost_equal(
            contract("r,rp,rq->pq", diags, made_u, made_u),
            contract("r,pr,qr->pq", diags, made_u, made_u),
        )
        vals, vecs = np.linalg.eigh(symm)
        vecs = vecs.astype(complex)
        vec_thetas, vec_diag_theta = extract_thetas(vecs)
        real_x = sp.linalg.logmh(vecs)
        faek_x = make_x_matrix(thetas=vec_thetas, diags=vec_diag_theta, n=4)
        np.testing.assert_array_almost_equal(real_x, faek_x)
        vec_u = make_unitary_im(vec_thetas, vec_diag_theta, 4)
        np.testing.assert_array_almost_equal(
            contract("r,pr,qr->pq", diags, u, u),
            contract("r,pr,qr->pq", vals, vec_u, vec_u),
        )

    @given(generate_symm_unitary_matrices(n=4))
    @settings(max_examples=20)
    def test_jl_unitary(self, u_symm):
        try:
            jl_extract_thetas(np.random.rand(4, 4))
        except RuntimeError:
            pass
        diags, u, symm = u_symm
        thetas, diag_thetas = jl_extract_thetas(u)
        jl_u = jl_make_u_im(thetas, diag_thetas, 4)
        np.testing.assert_array_almost_equal(jl_u, u)

    @given(H_2_LR())
    def test_unitary_lr(self, obt_tbt_frags_bl):
        frags: List[LRFragment]
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        for f in frags:
            x_matrix = make_x_matrix(
                thetas=f.thetas,
                diags=f.diag_thetas,
                n=4,
                imag=isinstance(f.diag_thetas, np.ndarray),
            )
            x_matrix_jl = (
                jl_make_x_im(t=f.thetas, d=f.diag_thetas, n=4)
                if isinstance(f.diag_thetas, np.ndarray)
                else jl_make_x(f.thetas, 4)
            )
            try:
                np.testing.assert_array_almost_equal(
                    x_matrix,
                    x_matrix_jl,
                )
            except:
                jl_compare_matrices(x_matrix, x_matrix_jl)
                self.fail()
            unitary = (
                make_unitary_im(f.thetas, f.diag_thetas, 4)
                if isinstance(f.diag_thetas, np.ndarray)
                else make_unitary(f.thetas, n=4, imag=False)
            )
            unitary_jl = (
                jl_make_u_im(f.thetas, f.diag_thetas, 4)
                if isinstance(f.diag_thetas, np.ndarray)
                else jl_make_u(f.thetas, 4)
            )
            try:
                np.testing.assert_array_almost_equal(
                    unitary,
                    unitary_jl,
                )
            except:
                jl_compare_matrices(unitary, unitary_jl)
                self.fail()
            self.assertEqual(
                f.operators,
                tbt2op(get_lr_fragment_tensor(f)),
            )

    @given(H_2_GFRO())
    def test_unitary_gfro(self, obt_tbt_frags_bl):
        frags: List[GFROFragment]
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        for f in frags:
            x_matrix = make_x_matrix(
                thetas=f.thetas,
                n=4,
            )
            x_matrix_jl = jl_make_x(f.thetas, 4)
            try:
                np.testing.assert_array_almost_equal(
                    x_matrix,
                    x_matrix_jl,
                )
            except:
                jl_compare_matrices(x_matrix, x_matrix_jl)
                self.fail()
            unitary = make_unitary(f.thetas, n=4, imag=False)
            unitary_jl = jl_make_u(f.thetas, 4)
            try:
                np.testing.assert_array_almost_equal(
                    unitary,
                    unitary_jl,
                )
            except:
                jl_compare_matrices(unitary, unitary_jl)
                self.fail()
            self.assertEqual(
                f.operators,
                tbt2op(make_fr_tensor(f.lambdas, f.thetas, 4)),
            )
