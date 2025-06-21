import unittest
from copy import copy
from functools import reduce
from typing import List

import numpy as np
import scipy as sp
from hypothesis import given, strategies as st, settings, HealthCheck
from openfermion import (
    FermionOperator,
    jordan_wigner,
)
from opt_einsum import contract

from d_types.fragment_types import GFROFragment, FluidCoeff
from min_part.f_3_ops import (
    obt2fluid,
    fluid_2tensor,
    static_2tensor,
)
from min_part.gfro_decomp import (
    gfro_decomp,
    extract_thetas,
    make_x_matrix,
    make_lambda_matrix,
)
from min_part.lr_decomp import make_unitary_im
from min_part.operators import (
    assert_number_operator_equality,
    collapse_to_number_operator,
)
from min_part.tensor import (
    get_n_body_tensor_chemist_ordering,
    obt2op,
    tbt2op,
)
from min_part.testing_utils.sim_tensor import (
    generate_symm_unitary_matrices,
)
from min_part.testing_utils.sim_molecules import H_2_GFRO

settings.register_profile("slow", deadline=None)
settings.load_profile("slow")


def tensors_equal(H_tbt: np.ndarray, gfro_h2_frags: List[GFROFragment], n: int):
    np.testing.assert_array_almost_equal(
        H_tbt,
        reduce(
            lambda a, b: a + b,
            [g.to_tensor() for g in gfro_h2_frags],
            np.zeros((n, n)),
        ),
    )


class FluidFragmentTest(unittest.TestCase):
    def operators_equal(self, H_tbt: np.ndarray, gfro_h2_frags: List[GFROFragment]):
        self.assertEqual(
            tbt2op(H_tbt),
            reduce(
                lambda a, b: a + b,
                [g.operators for g in gfro_h2_frags],
                FermionOperator(),
            ),
        )

    # == GFRO Tests ==
    # PASS
    # TODO: Do we need to filter out det(matrices) != 1
    @given(generate_symm_unitary_matrices(n=4))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.data_too_large])
    def test_obt_2_fluid(self, vals_vecs_mat):
        vals, vecs, obt = vals_vecs_mat
        V, U = np.linalg.eigh(obt)
        np.testing.assert_array_almost_equal(obt, obt.T)
        theta, diag = extract_thetas(U)
        X = make_x_matrix(thetas=theta, n=4, diags=diag, imag=True)
        U_x = sp.linalg.expm(X)
        np.testing.assert_array_almost_equal(U_x, U)
        a_fluid = obt2fluid(obt)
        np.testing.assert_array_almost_equal(a_fluid.diag_thetas, diag)
        np.testing.assert_array_almost_equal(a_fluid.thetas, theta)
        np.testing.assert_array_almost_equal(a_fluid.lambdas, V)
        np.testing.assert_array_almost_equal(a_fluid.to_tensor(), obt)
        np.testing.assert_array_almost_equal(
            get_n_body_tensor_chemist_ordering(a_fluid.to_op(), n=1, m=4), obt
        )

    # PASS
    @given(H_2_GFRO())
    @settings(max_examples=30)
    def test_fluid_gfro_tensor_to_op(self, obt_tbt_gfrofrag_bondlength):
        gfro_h2_frags: List[GFROFragment]
        H_obt, H_tbt, gfro_h2_frags, bond_legnth = obt_tbt_gfrofrag_bondlength
        for gfro_frag in gfro_h2_frags:
            prev_operators = copy(gfro_frag.operators)
            gfro_frag.to_fluid()
            self.assertEqual(prev_operators, gfro_frag.to_op())
        self.operators_equal(H_tbt, gfro_h2_frags)

    # PASS
    @given(H_2_GFRO())
    @settings(max_examples=5)
    def test_gfro_fluid_to_tensor(self, obt_tbt_gfrofrags_bl):
        gfro_h2_frags: List[GFROFragment]
        H_obt, H_tbt, gfro_h2_frags, bond_length = obt_tbt_gfrofrags_bl
        for gfro_frag in gfro_h2_frags:
            og_ten = get_n_body_tensor_chemist_ordering(gfro_frag.operators, n=2, m=4)
            gfro_frag.to_fluid()
            fluid_ten = fluid_2tensor(
                gfro_frag.fluid_parts.fluid_lambdas, gfro_frag.thetas
            ) + static_2tensor(gfro_frag)
            np.testing.assert_array_almost_equal(og_ten, fluid_ten)
        tensors_equal(H_tbt, gfro_h2_frags, H_tbt.shape[0])

    # PASS
    @given(H_2_GFRO())
    @settings(max_examples=5)
    def test_gfro_paritioning(self, obt_tbt_gfrofrags_bl):
        gfro_h2_frags: List[GFROFragment]
        H_obt, H_tbt, gfro_h2_frags, bond_length = obt_tbt_gfrofrags_bl
        self.operators_equal(H_tbt, gfro_h2_frags)
        for gfro_frag in gfro_h2_frags:
            og_tensor = get_n_body_tensor_chemist_ordering(
                gfro_frag.operators, n=2, m=4
            )
            gfro_frag.to_fluid()
            fluid_static_sum = (
                fluid_2tensor(gfro_frag.fluid_parts.fluid_lambdas / 4, gfro_frag.thetas)
                + fluid_2tensor(
                    gfro_frag.fluid_parts.fluid_lambdas / 4, gfro_frag.thetas
                )
                + fluid_2tensor(
                    gfro_frag.fluid_parts.fluid_lambdas / 2, gfro_frag.thetas
                )
                + static_2tensor(gfro_frag)
            )
            np.testing.assert_array_almost_equal(
                og_tensor,
                fluid_static_sum,
            )
        tensors_equal(H_tbt, gfro_h2_frags, H_tbt.shape[0])

    @given(
        st.lists(
            st.floats(-2, 2, allow_nan=False, allow_infinity=False),
            max_size=10,
            min_size=10,
        ),
    )
    def test_get_one_body_parts(self, lambdas):  # PASS
        diags = np.diag(make_lambda_matrix(np.array(lambdas), n=4))
        gfro_frag = GFROFragment(
            lambdas=np.array(lambdas), thetas=[], operators=FermionOperator()
        )
        np.testing.assert_array_equal(diags, gfro_frag.get_ob_lambdas())

    @given(H_2_GFRO())
    def test_moving_coeffs_obf_same_dim_matrices(
        self, H_obt_H_tbt_gfro_h2_frags_lr_h2_frags
    ):
        H_obt, H_tbt, gfro_h2_frags, lr_h2_frags = H_obt_H_tbt_gfro_h2_frags_lr_h2_frags
        coeff = (
            np.random.uniform(low=H_obt[0][0], high=0, size=1)
            if H_obt[0][0] < 0
            else np.random.uniform(low=0, high=H_obt[0][0], size=1)
        )
        ob_f = obt2fluid(H_obt)
        ob_f.fluid_lambdas.append((0, FluidCoeff(coeff=coeff, thetas=ob_f.thetas)))
        u = make_unitary_im(thetas=ob_f.thetas, diags=ob_f.diag_thetas, n=4)
        frag_ten = contract("r,rp,rq->pq", [coeff, 0, 0, 0], u, u)
        np.testing.assert_array_almost_equal(H_obt, ob_f.to_tensor() - frag_ten)

    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
            lambda n: n != 0
        ),
        generate_symm_unitary_matrices(n=4),
    )
    def test_0_case_dif_dims_matrices(self, coeff, vals, vec, a):
        b = np.zeros((4, 4, 4, 4))
        b[0, 0, 0, 0] = coeff
        gfro_frags = gfro_decomp(b)
        if len(gfro_frags) != 0:
            first_frag_ops = copy(gfro_frags[0].operators)
            from_frag = gfro_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            from_frag.move2frag(to=fluid_1, orb=0, coeff=0, mutate=True)
            self.assertEqual(first_frag_ops, from_frag.to_op())
            self.assertEqual(
                tbt2op(b),
                from_frag.to_op()
                + reduce(
                    lambda a, b: a + b,
                    [g.operators for g in gfro_frags[1:]],
                    FermionOperator(),
                ),
            )
            np.testing.assert_array_almost_equal(fluid_1.to_tensor(), a)
            self.assertEqual(fluid_1.to_op(), obt2op(a))
            self.assertEqual(fluid_1.to_op() + fluid_1.to_op(), obt2op(a) + tbt2op(b))

    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
            lambda n: n != 0
        ),
        generate_symm_unitary_matrices(n=4),
    )
    def test_entire_coeff_case_dif_dims_matrices(self, coeff, vals_vecs_symm):
        n = 4
        vals, vec, a = vals_vecs_symm
        b = np.zeros((n, n, n, n))
        b[0, 0, 0, 0] = coeff
        gfro_frags = gfro_decomp(b)
        if len(gfro_frags) > 0:
            old_lambda = gfro_frags[0].lambdas[0]
            from_frag = gfro_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            from_frag.move2frag(to=fluid_1, orb=0, coeff=coeff, mutate=True)
            self.assertEqual(from_frag.fluid_parts.fluid_lambdas[0], old_lambda - coeff)
            fluid_total = from_frag.to_op() + fluid_1.to_op()
            og_total = obt2op(a) + tbt2op(b)
            if assert_number_operator_equality(fluid_total, og_total):
                self.assertTrue(assert_number_operator_equality(fluid_total, og_total))
            else:
                self.assertEqual(jordan_wigner(fluid_total), jordan_wigner(og_total))
                self.assertEqual(
                    collapse_to_number_operator(fluid_total),
                    collapse_to_number_operator(og_total),
                )

    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
            lambda n: n != 0
        ),
        generate_symm_unitary_matrices(n=4),
    )
    def test_dif_thetas_dif_dims_matrices(self, coeff, vals, vec, fake_obt):
        tbt_op = (
            FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (1, 1), (1, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (1, 1), (1, 0)))
        )
        tbt_ten = get_n_body_tensor_chemist_ordering(tbt_op, n=2, m=4)
        gfro_frag = gfro_decomp(tbt_ten)[0]
        total_op = obt2op(fake_obt) + gfro_frag.operators
        # == fluid begins ==
        fake_ob_fluid = obt2fluid(fake_obt)
        fake_tb_fluid = gfro_frag.to_fluid()
        # == move 0 check ==
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=0, mutate=True)
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=1, mutate=True)
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=2, mutate=True)
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=3, mutate=True)
        self.assertEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
        # == move 1/2 coeff check ==
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=coeff / 2, orb=1, mutate=True)
        self.assertEqual(
            jordan_wigner(total_op),
            jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
        )
        self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
        # == move all coeff check ==
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=coeff / 2, orb=1, mutate=True)
        self.assertEqual(
            jordan_wigner(total_op),
            jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
        )
        self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
        fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=coeff, orb=0, mutate=True)
        self.assertEqual(
            jordan_wigner(total_op),
            jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
        )
        self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
        self.assertAlmostEqual(fake_tb_fluid.fluid_parts.fluid_lambdas[0], 0)
        self.assertAlmostEqual(fake_tb_fluid.fluid_parts.fluid_lambdas[1], 0)
        self.assertEqual(fake_ob_fluid.fluid_lambdas[5][1].coeff, coeff / 2)
        self.assertEqual(fake_ob_fluid.fluid_lambdas[5][0], 1)
        self.assertEqual(fake_ob_fluid.fluid_lambdas[6][1].coeff, coeff)
        self.assertEqual(fake_ob_fluid.fluid_lambdas[6][0], 0)

    @given(H_2_GFRO())
    def test_mutate_each_frag_gfro(self, H_obt_H_tbt_gfro_h2_frags_lr_h2_frags):
        gfro_h2_frags: List[GFROFragment]
        H_obt, H_tbt, gfro_h2_frags, bond_legnth = H_obt_H_tbt_gfro_h2_frags_lr_h2_frags

    # == LR Tests Begin ==
    def test_convert_lr_2b_to_f3(self):
        pass

    def test_move_from_2b_2_1b(self):
        pass

    def test_move_from_2b_2_1b_multiple(self):
        pass

    def test_rediag_1b(self):
        pass

    def test_add_b_eq_a_coeff_lr(self):
        pass

    def test_add_b_less_a_coeff_lr(self):
        pass

    def test_add_b_more_a_coeff_lr(self):
        pass
