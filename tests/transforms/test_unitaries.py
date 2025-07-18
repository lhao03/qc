import unittest

import numpy as np
import scipy as sp
from hypothesis import given, example, settings
from opt_einsum import contract

from d_types.unitary_type import (
    extract_thetas,
    make_unitary_im,
    jl_extract_thetas,
    make_x_matrix,
    jl_make_u_im,
    Unitary,
)

from tests.utils.sim_tensor import (
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

    @given(generate_symm_unitary_matrices(n=4))
    @settings(max_examples=3)
    def test_spin2spac(self, u_symm):
        _, u, _ = u_symm
        spin_u = Unitary.deconstruct_unitary(u)
        spac_u = Unitary.deconstruct_unitary(u).spin2spac()
        np.testing.assert_array_equal(
            spin_u.make_unitary_matrix(), spac_u.spac2spin().make_unitary_matrix()
        )
