import unittest
from copy import copy
from functools import reduce
from typing import List, Tuple

import numpy as np
import scipy as sp
from hypothesis import given, strategies as st, settings, HealthCheck
from openfermion import (
    FermionOperator,
    jordan_wigner,
    qubit_operator_sparse,
)
from opt_einsum import contract

from d_types.fragment_types import (
    GFROFragment,
    FluidCoeff,
    LRFragment,
    FermionicFragment,
)
from min_part.f3_opers import (
    get_diag_idx,
    obt2fluid,
    fluid_2tensor,
    static_2tensor,
    make_obp_tensor,
    fluid_lr_2tensor,
    make_lambdas,
    lambdas_from_fluid_parts,
)

from min_part.gfro_decomp import (
    gfro_decomp,
    make_fr_tensor,
)
from min_part.lr_decomp import (
    get_lr_fragment_tensor_from_parts,
    lr_decomp,
    get_lr_fragment_tensor_from_lambda,
    lr_fragment_occ_from_lambdas,
)
from min_part.operators import (
    assert_number_operator_equality,
    collapse_to_number_operator,
    subspace_restriction,
)
from min_part.tensor import (
    get_n_body_tensor_chemist_ordering,
    obt2op,
    tbt2op,
    make_lambda_matrix,
    extract_lambdas,
)
from d_types.helper_types import extract_thetas, make_unitary_im

from tests.utils.sim_molecules import (
    H_2_GFRO,
    H_2_LR,
    specific_gfro_decomp,
    specific_lr_decomp,
)
from tests.utils.sim_tensor import generate_symm_unitary_matrices, make_tensors_h2

settings.register_profile("slow", deadline=None, print_blob=True)
settings.load_profile("slow")


def tensors_equal(H_tbt: np.ndarray, frags: List[FermionicFragment], n: int):
    np.testing.assert_array_almost_equal(
        H_tbt,
        reduce(
            lambda a, b: a + b,
            [g.to_tensor() for g in frags],
            np.zeros((n, n)),
        ),
    )


def add_fluid_diags_to_static(f, curr_fluid_lambdas: List[Tuple[int, FluidCoeff]]):
    n = f.fluid_parts.fluid_lambdas.size
    copy_lambda = copy(f.fluid_parts.static_lambdas)
    copy_lambda.setflags(write=True)
    for i in range(n):
        idx = get_diag_idx(i, n)
        copy_lambda[idx] = f.fluid_parts.fluid_lambdas[i]
        for o, c in curr_fluid_lambdas:
            if o == i:
                copy_lambda[idx] += c.coeff
    return copy_lambda


class FluidFragmentTest(unittest.TestCase):
    def operators_equal(self, H_tbt: np.ndarray, frags: List[FermionicFragment]):
        self.assertEqual(
            tbt2op(H_tbt),
            reduce(
                lambda a, b: a + b,
                [g.operators for g in frags],
                FermionOperator(),
            ),
        )

    def test_diag(self):
        zeroth = get_diag_idx(0, 4)
        first = get_diag_idx(1, 4)
        second = get_diag_idx(2, 4)
        third = get_diag_idx(3, 4)
        self.assertEqual(zeroth, 0)
        self.assertEqual(first, 4)
        self.assertEqual(second, 7)
        self.assertEqual(third, 9)

    # == GFRO Tests ==
    # PASS
    @given(generate_symm_unitary_matrices(n=4))
    @settings(max_examples=3, suppress_health_check=[HealthCheck.data_too_large])
    def test_obt_2_fluid(self, vals_vecs_mat):
        vals, vecs, obt = vals_vecs_mat
        a_fluid = obt2fluid(obt)
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

    # PASS
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

    # PASS
    @given(H_2_GFRO(tol=1e-7), generate_symm_unitary_matrices(n=4), st.integers(0, 3))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.data_too_large])
    def test_moving_coeffs_obf_same_dim_matrices(
        self,
        obt_plusextra,  # =specfic_gfro_decomp(1)
        vals_vecs_symm,  # =rand_symm_matr(4)
        orb,  # =0
    ):
        H_obt, _, _, _ = obt_plusextra
        eigs, U, symm_mat = vals_vecs_symm
        coeff = eigs[orb] / 2
        ob_f = obt2fluid(H_obt)
        thetas, diags = extract_thetas(U)
        ob_f.fluid_lambdas.append(
            (orb, FluidCoeff(coeff=coeff, thetas=thetas, diag_thetas=diags))
        )
        tensor_total = H_obt + symm_mat
        operator_total = obt2op(tensor_total)
        eigs[orb] -= coeff
        u = make_unitary_im(thetas, diags, 4)
        np.testing.assert_array_almost_equal(u, U)
        frag_ten = contract("r,rp,rq->pq", eigs, u, u)  # TODO: prev contract pattern
        fluid_tensor = ob_f.to_tensor() + frag_ten
        fluid_operator = obt2op(fluid_tensor)
        np.testing.assert_array_almost_equal(tensor_total, fluid_tensor)
        self.assertEqual(operator_total, fluid_operator)

    # PASS: after increasing GFRO Decomp Accuracy
    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
            lambda n: n != 0
        ),
        generate_symm_unitary_matrices(n=4),
    )
    @settings(max_examples=10)
    def test_0_case_dif_dims_matrices(
        self,
        coeff,  # =0.1,
        vals_vec_a,  # =rand_symm_matr(4)
    ):
        vals, vec, a = vals_vec_a
        b = np.zeros((4, 4, 4, 4))
        b[0, 0, 0, 0] = coeff
        gfro_frags = gfro_decomp(b, threshold=1e-10, debug=True)
        self.operators_equal(b, gfro_frags)
        if len(gfro_frags) == 1:
            first_frag_ops = copy(gfro_frags[0].operators)
            from_frag = gfro_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            self.assertEqual(fluid_1.to_op(), obt2op(a))
            from_frag.move2frag(to=fluid_1, orb=0, coeff=0, mutate=True)
            self.assertEqual(first_frag_ops, from_frag.to_op())
            np.testing.assert_array_almost_equal(fluid_1.to_tensor(), a)
            self.assertEqual(fluid_1.to_op(), obt2op(a))
            self.assertEqual(fluid_1.to_op() + from_frag.to_op(), obt2op(a) + tbt2op(b))

    @given(
        st.floats(0, 2, allow_nan=False, allow_infinity=False).filter(lambda n: n != 0),
        generate_symm_unitary_matrices(n=4),
    )  # PASS
    @settings(max_examples=10)
    def test_entire_coeff_case_dif_dims_matrices(self, coeff, vals_vecs_symm):
        n = 4
        vals, vec, a = vals_vecs_symm
        b = np.zeros((n, n, n, n))
        b[0, 0, 0, 0] = coeff
        gfro_frags = gfro_decomp(b, threshold=1e-10, debug=True)
        if len(gfro_frags) == 1:
            old_ops = gfro_frags[0].operators
            old_lambda = gfro_frags[0].lambdas[0]
            from_frag = gfro_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            from_frag.move2frag(to=fluid_1, orb=0, coeff=coeff, mutate=True)
            self.assertEqual(from_frag.fluid_parts.fluid_lambdas[0], old_lambda - coeff)
            fluid_total = from_frag.to_op() + fluid_1.to_op()
            og_total = obt2op(a) + old_ops
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
        st.integers(0, 3),
    )
    @settings(max_examples=5)  # PASS
    def test_dif_thetas_dif_dims_matrices(self, coeff, vals_vec_symm, orb):
        vals, vec, fake_obt = vals_vec_symm
        tbt_op = (
            FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (1, 1), (1, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (1, 1), (1, 0)))
        )
        tbt_ten = get_n_body_tensor_chemist_ordering(tbt_op, n=2, m=4)
        decomp = gfro_decomp(tbt_ten, threshold=1e-8, debug=True)
        if len(decomp) > 0:
            gfro_frag = decomp[0]
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
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff / 2, orb=orb, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )
            self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
            # == move all coeff check ==
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff / 2, orb=orb, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )
            self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff, orb=orb % 3, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )
            self.assertNotEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())

    # @reproduce_failure("6.135.7", b"AEEKKEAAukijnlzr")
    # @given(st.integers(1, 10), H_2_GFRO())
    # @settings(max_examples=2)
    def test_mutate_each_frag_gfro(
        self,
        partition=1,
        obt_tbt_frags_bl=specific_gfro_decomp(0.8),
    ):
        frags: List[GFROFragment]
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        obt_f = obt2fluid(H_obt)
        prev_og_op = []
        h_pq = H_obt
        for n, f in enumerate(frags):
            f.to_fluid()
            prev_og_op.append(f.operators)
            fake_moved = np.zeros((10,))
            # check conversion to fluid keeps operator sum same
            np.testing.assert_array_almost_equal(
                get_n_body_tensor_chemist_ordering(f.operators, 2, 4),
                make_fr_tensor(f.lambdas, f.thetas, 4),
            )
            curr_fluid_lambdas: List[Tuple[int, FluidCoeff]] = []
            for i in range(4):
                to_move = f.fluid_parts.fluid_lambdas[i] / partition
                for p in range(1, partition + 1):
                    fake_moved[get_diag_idx(i, 4)] = to_move * p
                    fake_remaining = f.lambdas - fake_moved
                    np.testing.assert_array_almost_equal(
                        fake_moved + fake_remaining, f.lambdas
                    )
                    f.move2frag(to=obt_f, orb=i, coeff=to_move, mutate=True)
                    print(
                        f"Checking: moving {to_move} from {i}th spin orbital for frag {n}."
                    )
                    print(f"fluid: {f.fluid_parts.fluid_lambdas}")
                    curr_fluid_lambdas.append(obt_f.fluid_lambdas[-1])
                    # check current partition of fluid and static lambdas maintains frag operator sum
                    np.testing.assert_array_almost_equal(
                        get_n_body_tensor_chemist_ordering(prev_og_op[-1], 2, 4),
                        make_fr_tensor(fake_moved, f.thetas, 4)
                        + make_fr_tensor(fake_remaining, f.thetas, 4),
                    )
                    # check moved fluid portion is equal for current frag
                    t, o = (
                        jordan_wigner(tbt2op(make_fr_tensor(fake_moved, f.thetas, 4))),
                        jordan_wigner(
                            obt2op(
                                reduce(
                                    lambda a, b: a + b,
                                    [
                                        make_obp_tensor(l[1], 4, l[0])
                                        for l in curr_fluid_lambdas
                                    ],
                                )
                            )
                        ),
                    )

                    try:
                        self.assertEqual(
                            t, o
                        )  # TODO: ??? small floats might cause operator sum issues
                    except:
                        print("Failed the check at JW stage")
                    try:
                        # check obt portion is equal to moved over fluid parts
                        self.assertTrue(
                            jordan_wigner(
                                obt2op(h_pq)
                                + tbt2op(make_fr_tensor(fake_moved, f.thetas, 4))
                            ),
                            jordan_wigner(obt_f.to_op()),
                        )
                        print("Passed current obt is equal to h_pq + moved fluid")
                    except:
                        print("Failed current obt is equal to h_pq + moved fluid")
                    try:
                        # check remaining tbt is equal to current static amount
                        self.assertTrue(
                            f.to_op(),
                            tbt2op(make_fr_tensor(fake_remaining, f.thetas, 4)),
                        )
                        print("Passed remaining tbt is equal to tbt to op!")
                    except:
                        print("Failed remaining tbt is equal to tbt to op!")
            h_pq = obt_f.to_tensor()
        self.assertEqual(
            jordan_wigner(obt2op(H_obt) + tbt2op(H_tbt)),
            jordan_wigner(
                obt_f.to_op()
                + reduce(
                    lambda a, b: a + b,
                    [f.to_op() for f in frags],
                    FermionOperator(),
                )
            ),
        )

    def test_gfro_expectation_val(self, obt_tbt_frags_bl=specific_gfro_decomp(0.8)):
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        expectation_values = [
            f.get_expectation_value(num_spin_orbs=4, expected_e=2) for f in frags
        ]
        for f in frags:
            f.to_fluid(performant=False)
        fluid_expectation_values = [
            f.get_expectation_value(num_spin_orbs=4, expected_e=2) for f in frags
        ]
        for occ_e_occ_e in zip(expectation_values, fluid_expectation_values):
            og = occ_e_occ_e[0]
            f = occ_e_occ_e[1]
            occs = og[0]
            f_occs = f[0]
            es = og[1]
            f_es = f[1]
            for i in range(6):
                self.assertEqual(occs[i], f_occs[i])
                self.assertEqual(es[i], f_es[i])

    def test_rediag_1b(self):
        pass

    # == LR Tests Begin ==
    @given(H_2_LR())
    @settings(max_examples=30)
    def test_fluid_lr_tensor_to_op(self, obt_tbt_lr_bl):
        lr_fs: List[LRFragment]
        H_obt, H_tbt, lr_fs, bl = obt_tbt_lr_bl
        for f in lr_fs:
            prev_operators = copy(f.operators)
            f.to_fluid()
            self.assertEqual(prev_operators, f.to_op())
        self.operators_equal(H_tbt, lr_fs)

    # PASS
    @given(H_2_LR())
    @settings(max_examples=30)
    def test_lr_fluid_to_tensor(self, obt_tbt_lr_bl):
        lr_frags: List[LRFragment]
        H_obt, H_tbt, lr_frags, bl = obt_tbt_lr_bl
        for f in lr_frags:
            og_ten = get_n_body_tensor_chemist_ordering(f.operators, n=2, m=4)
            f.to_fluid()
            fluid_ten = fluid_lr_2tensor(f)
            np.testing.assert_array_almost_equal(og_ten, fluid_ten)
        tensors_equal(H_tbt, lr_frags, H_tbt.shape[0])

    @given(
        st.floats(0, 2, allow_nan=False, allow_infinity=False).filter(lambda n: n != 0),
        generate_symm_unitary_matrices(n=4),
    )
    @settings(max_examples=10)
    def test_entire_coeff_case_dif_dims_matrices_lr(self, coeff, vals_vecs_symm):
        print(f"now trying: {coeff}")
        n = 4
        vals, vec, a = vals_vecs_symm
        b = np.zeros((n, n, n, n))
        b[0, 0, 0, 0] = coeff
        lr_frags = lr_decomp(b)
        if len(lr_frags) == 1:
            old_ops = lr_frags[0].operators
            from_frag = lr_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            from_frag.move2frag(to=fluid_1, orb=0, coeff=coeff, mutate=True)
            fluid_total = from_frag.to_op() + fluid_1.to_op()
            og_total = obt2op(a) + old_ops
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
        st.integers(0, 3),
    )
    @settings(max_examples=10)
    def test_dif_thetas_dif_dims_matrices_lr(self, coeff, vals_vec_symm, orb):
        vals, vec, fake_obt = vals_vec_symm
        tbt_op = (
            FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (1, 1), (1, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (1, 1), (1, 0)))
        )
        tbt_ten = get_n_body_tensor_chemist_ordering(tbt_op, n=2, m=4)
        decomp = lr_decomp(tbt_ten)
        if len(decomp) > 0:
            frag = decomp[0]
            total_op = obt2op(fake_obt) + frag.operators
            # == fluid begins ==
            fake_ob_fluid = obt2fluid(fake_obt)
            fake_tb_fluid = frag.to_fluid()
            # == move 0 check ==
            fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=0, mutate=True)
            fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=1, mutate=True)
            fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=2, mutate=True)
            fake_tb_fluid.move2frag(to=fake_ob_fluid, coeff=0, orb=3, mutate=True)
            self.assertEqual(total_op, fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
            # == move 1/2 coeff check ==
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff / 2, orb=orb, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )
            # == move all coeff check ==
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff / 2, orb=orb, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )
            fake_tb_fluid.move2frag(
                to=fake_ob_fluid, coeff=coeff, orb=orb % 3, mutate=True
            )
            self.assertEqual(
                jordan_wigner(total_op),
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op()),
            )

    # @given(st.integers(1, 20), H_2_LR())
    # @settings(max_examples=1)
    def test_mutate_each_frag_lr(
        self, partition=10, obt_tbt_frags_bl=specific_lr_decomp(2.0909588606551686)
    ):
        frags: List[LRFragment]
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        obt_f = obt2fluid(H_obt)
        prev_og_op = []
        h_pq = H_obt
        for n, f in enumerate(frags):
            f.to_fluid()
            prev_og_op.append(f.operators)
            fake_moved = np.zeros((10,))
            # check conversion to fluid keeps operator sum same
            np.testing.assert_array_almost_equal(
                get_n_body_tensor_chemist_ordering(f.operators, 2, 4),
                get_lr_fragment_tensor_from_parts(
                    outer_coeff=f.outer_coeff,
                    coeffs=f.coeffs,
                    thetas=f.thetas,
                    diags_thetas=f.diag_thetas,
                ),
            )
            f_lambdas = extract_lambdas(f.outer_coeff * f.coeffs @ f.coeffs.T, 4)
            curr_fluid_lambdas: List[Tuple[int, FluidCoeff]] = []
            for i in range(4):
                to_move = f.fluid_parts.fluid_lambdas[i] / partition
                for p in range(1, partition + 1):
                    fake_moved[get_diag_idx(i, 4)] = to_move * p
                    fake_remaining = f_lambdas - fake_moved
                    np.testing.assert_array_almost_equal(
                        fake_moved + fake_remaining,
                        add_fluid_diags_to_static(f, curr_fluid_lambdas),
                    )
                    f.move2frag(to=obt_f, orb=i, coeff=to_move, mutate=True)
                    print(
                        f"Checking: moving {to_move} from {i}th spin orbital for frag {n}."
                    )
                    print(f"fluid: {f.fluid_parts.fluid_lambdas}")
                    curr_fluid_lambdas.append(obt_f.fluid_lambdas[-1])
                    # check current partition of fluid and static lambdas maintains frag operator sum
                    np.testing.assert_array_almost_equal(
                        get_n_body_tensor_chemist_ordering(prev_og_op[-1], 2, 4),
                        get_lr_fragment_tensor_from_lambda(
                            fake_moved, f.thetas, f.diag_thetas, n=4
                        )
                        + get_lr_fragment_tensor_from_lambda(
                            fake_remaining, f.thetas, f.diag_thetas, n=4
                        ),
                    )
                    np.testing.assert_array_almost_equal(
                        make_lambdas(curr_fluid_lambdas, 4), fake_moved
                    )
                    # check moved fluid portion is equal for current frag
                    t, o = (
                        jordan_wigner(
                            tbt2op(
                                get_lr_fragment_tensor_from_lambda(
                                    fake_moved, f.thetas, f.diag_thetas, 4
                                )
                            )
                        ),
                        jordan_wigner(
                            obt2op(
                                reduce(
                                    lambda a, b: a + b,
                                    [
                                        make_obp_tensor(l[1], 4, l[0])
                                        for l in curr_fluid_lambdas
                                    ],
                                )
                            )
                        ),
                    )

                    try:
                        self.assertEqual(
                            t, o
                        )  # TODO: ??? small floats might cause operator sum issues
                        print("Passed the check at JW stage")
                    except:
                        print("Failed the check at JW stage")
                    try:
                        # check obt portion is equal to moved over fluid parts
                        self.assertTrue(
                            jordan_wigner(
                                obt2op(h_pq)
                                + tbt2op(
                                    get_lr_fragment_tensor_from_lambda(
                                        fake_moved, f.thetas, f.diag_thetas, 4
                                    )
                                )
                            ),
                            jordan_wigner(obt_f.to_op()),
                        )
                        print("Passed current obt is equal to h_pq + moved fluid")
                    except:
                        print("Failed current obt is equal to h_pq + moved fluid")
                    try:
                        # check remaining tbt is equal to current static amount
                        self.assertTrue(
                            f.to_op(),
                            tbt2op(
                                get_lr_fragment_tensor_from_lambda(
                                    fake_remaining, f.thetas, f.diag_thetas, 4
                                )
                            ),
                        )
                        print("Passed remaining tbt is equal to tbt to op!")
                    except:
                        print("Failed remaining tbt is equal to tbt to op!")
            h_pq = obt_f.to_tensor()
        self.assertEqual(
            jordan_wigner(obt2op(H_obt) + tbt2op(H_tbt)),
            jordan_wigner(
                obt_f.to_op()
                + reduce(
                    lambda a, b: a + b,
                    [f.to_op() for f in frags],
                    FermionOperator(),
                )
            ),
        )

    def test_fluid_lr_h2_occs(self):
        H_obt, H_tbt, frags, bl = specific_lr_decomp(2.0909588606551686)
        n = H_tbt.shape[0]
        for f in frags:
            f.to_fluid()
            diag_eigenvalues, diag_eigenvectors = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(f.operators)).toarray()
            )
            lambdas = lambdas_from_fluid_parts(f.fluid_parts)
            occupations, eigenvalues = lr_fragment_occ_from_lambdas(
                lambdas, num_spin_orbs=n
            )
            self.assertTrue(
                np.allclose(np.sort(diag_eigenvalues), np.sort(eigenvalues))
            )

    def test_projection_operator_lr(self):
        H_const, H_obt, H_tbt = make_tensors_h2(0.8)
        frags = lr_decomp(H_tbt)
        obt_e, vecs = np.linalg.eigh(
            subspace_restriction(H_const + obt2op(H_obt), 4, num_elecs=2).toarray()
        )
        tbt_e = 0
        for f in frags:
            tbt_es, vecs = np.linalg.eigh(
                subspace_restriction(f.operators, 4, num_elecs=2).toarray()
            )
            tbt_e += min(tbt_es)
        total_e, vecs = np.linalg.eigh(
            subspace_restriction(
                H_const + obt2op(H_obt) + tbt2op(H_tbt), 4, num_elecs=2
            ).toarray()
        )
        print(min(total_e))
        print(min(obt_e) + tbt_e)

    def test_subspace_proj_vs_occ_energies(
        self, partition=4, obt_tbt_frags_bl=specific_gfro_decomp(2.0909588606551686)
    ):
        frags: List[GFROFragment]
        H_obt, H_tbt, frags, bl = obt_tbt_frags_bl
        obt_f = obt2fluid(H_obt)
        for n, f in enumerate(frags):
            f.to_fluid()
            for i in range(4):
                to_move = f.fluid_parts.fluid_lambdas[i] / partition
                for p in range(1, partition + 1):
                    f.move2frag(to=obt_f, orb=i, coeff=to_move, mutate=True)
                    occs, energy = f.get_expectation_value(
                        num_spin_orbs=4, expected_e=2
                    )
                    # eigs, vecs = np.linalg.eigh(
                    #     subspace_projection_operator(
                    #         f.to_op(), n_spin_orbs=4, num_elecs=2
                    #     ).toarray()
                    # )
                    eigs, vecs = np.linalg.eigh(
                        qubit_operator_sparse(
                            jordan_wigner(f.to_op()),
                        ).toarray()
                    )
                    min_energy_in_eigs = False
                    min_occ_energy = min(energy)
                    for e in eigs:
                        if np.isclose(e, min_occ_energy):
                            min_energy_in_eigs = True
                    # self.assertTrue(min_energy_in_eigs)
        self.assertEqual(
            jordan_wigner(obt2op(H_obt) + tbt2op(H_tbt)),
            jordan_wigner(
                obt_f.to_op()
                + reduce(
                    lambda a, b: a + b,
                    [f.to_op() for f in frags],
                    FermionOperator(),
                )
            ),
        )
