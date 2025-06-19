import random
import unittest
from copy import copy
from functools import reduce

import numpy as np
import scipy as sp
from hypothesis import given, strategies as st, settings
from openfermion import (
    count_qubits,
    FermionOperator,
    jordan_wigner,
    qubit_operator_sparse,
)
from opt_einsum import contract

from d_types.fragment_types import GFROFragment, OneBodyFragment, FluidCoeff
from min_part.f_3_ops import (
    obt2fluid,
    gfro2fluid,
    fluid_ob2op,
)
from min_part.gfro_decomp import (
    gfro_decomp,
    make_unitary,
    make_fr_tensor_from_u,
    make_fr_tensor,
    extract_thetas,
    make_x_matrix,
    make_lambda_matrix,
)
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.lr_decomp import make_unitary_im
from min_part.molecules import mol_h2
from min_part.operators import (
    assert_number_operator_equality,
    collapse_to_number_operator,
)
from min_part.tensor import get_n_body_tensor, obt2op, tbt2op, get_chem_tensors

settings.register_profile("slow", deadline=None)
settings.load_profile("slow")


class FluidFragmentTest(unittest.TestCase):
    bond_length = 0.80
    mol = mol_h2(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
    H_ob_op = obt2op(H_obt)
    H_tb_op = tbt2op(H_tbt)

    def setUp(self):
        self.gfro_h2_frags = 0  # gfro_decomp(self.H_tbt)
        self.lr_h2_frags = 0  # lr_decomp(self.H_tbt)

    @given(
        st.lists(
            st.floats(-2, 2, allow_nan=False, allow_infinity=False),
            max_size=10,
            min_size=10,
        ),
    )
    def test_obt_2_fluid(self, lambdas):
        a = make_lambda_matrix(np.array(lambdas), n=4)
        V, U = np.linalg.eigh(a)
        np.testing.assert_array_equal(a, a.T)
        theta, diag = extract_thetas(U)
        X = make_x_matrix(thetas=theta, n=4, diags=diag, imag=True)
        U_x = sp.linalg.expm(X)
        np.testing.assert_array_almost_equal(U_x, U)
        a_fluid = obt2fluid(a)
        np.testing.assert_array_almost_equal(a_fluid.diag_thetas, diag)
        np.testing.assert_array_almost_equal(a_fluid.thetas, theta)
        np.testing.assert_array_almost_equal(a_fluid.lambdas, V)
        np.testing.assert_array_almost_equal(a_fluid.to_tensor(), a)
        np.testing.assert_array_almost_equal(
            get_n_body_tensor(a_fluid.to_op(), n=1, m=4), a
        )

    def test_gfro_paritioning(self):
        pass

    def test_gfro_fluid_to_tensor(self):
        pass

    def test_fluid_gfro_tensor_to_op(self):
        pass

    def test_get_one_body_parts(self):  # TODO: ask help to fix
        n = 5
        m = (n * (n + 1)) // 2
        fake_h2 = np.random.rand(m)
        diags = [fake_h2[0], fake_h2[5], fake_h2[9], fake_h2[12], fake_h2[14]]
        gfro_frag = GFROFragment(
            lambdas=np.array(fake_h2), thetas=[], operators=FermionOperator()
        )
        np.testing.assert_array_equal(diags, gfro_frag.get_ob_lambdas())

    @given(st.floats(H_obt[0][0].real, 0))
    def test_moving_coeffs_obf_same_dim_matrices(self, coeff):
        ob_f = obt2fluid(self.H_obt)
        ob_f.fluid_lambdas.append((0, FluidCoeff(coeff=coeff, thetas=ob_f.thetas)))
        u = make_unitary_im(thetas=ob_f.thetas, diags=ob_f.diag_thetas, n=4)
        frag_ten = contract("r,rp,rq->pq", [coeff, 0, 0, 0], u, u)
        np.testing.assert_array_almost_equal(self.H_obt, ob_f.to_tensor() - frag_ten)

    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
            lambda n: n != 0
        ),
        st.lists(
            st.floats(-2, 2, allow_nan=False, allow_infinity=False),
            max_size=4,
            min_size=4,
        ),
    )
    def test_0_case_dif_dims_matrices(self, coeff, ob):
        a = np.diagflat(ob)
        b = np.zeros((4, 4, 4, 4))
        b[0, 0, 0, 0] = coeff
        gfro_frags = gfro_decomp(b)
        if len(gfro_frags) != 0:
            tensor = make_fr_tensor(gfro_frags[0].lambdas, gfro_frags[0].thetas, 4)
            self.assertEqual(tbt2op(b), tbt2op(tensor))
            first_frag_ops = gfro_frags[0].operators
            from_frag = gfro_frags[0].to_fluid()
            fluid_1 = obt2fluid(a)
            tb_f: GFROFragment
            ob_f: OneBodyFragment
            tb_f, ob_f = from_frag.move2frag(to=fluid_1, orb=0, coeff=0, mutate=True)
            self.assertAlmostEqual(
                gfro_frags[0].lambdas[0],
                tb_f.lambdas[0],
                from_frag.fluid_parts.fluid_lambdas[0],
            )
            np.testing.assert_array_equal(from_frag.thetas, gfro_frags[0].thetas)
            self.assertEqual(first_frag_ops, tb_f.to_op())
            self.assertEqual(
                tbt2op(b),
                tb_f.to_op()
                + reduce(
                    lambda a, b: a + b,
                    [g.operators for g in gfro_frags[1:]],
                    FermionOperator(),
                ),
            )
            np.testing.assert_array_almost_equal(ob_f.to_tensor(), a)
            self.assertEqual(ob_f.to_op(), obt2op(a))
            self.assertEqual(tb_f.to_op() + ob_f.to_op(), obt2op(a) + tbt2op(b))

    def test_entire_coeff_case_dif_dims_matrices(self):
        n = 4
        a = np.array(
            [[0.3, 0, 0, 0], [0, 0.12, 0, 0], [0, 0, 0.32, 0], [0, 0, 0, 0.43]]
        )
        b = np.zeros((n, n, n, n))
        b[0, 0, 0, 0] = 0.31
        gfro_frags = gfro_decomp(b)
        old_lambda = gfro_frags[0].lambdas[0]
        self.assertEqual(
            obt2op(a) + tbt2op(b),
            reduce(
                lambda a, b: a + b, [g.operators for g in gfro_frags], FermionOperator()
            )
            + obt2op(a),
        )
        from_frag = gfro_frags[0].to_fluid()
        coeff = 0.31
        fluid_1 = obt2fluid(a)
        np.testing.assert_array_equal(from_frag.thetas, gfro_frags[0].thetas)
        np.testing.assert_array_equal(from_frag.lambdas, gfro_frags[0].lambdas)
        self.assertEqual(old_lambda, from_frag.fluid_parts.fluid_lambdas[0])
        self.assertAlmostEqual(0, gfro_frags[0].fluid_parts.fluid_lambdas[1])
        self.assertAlmostEqual(0, gfro_frags[0].fluid_parts.fluid_lambdas[2])
        self.assertAlmostEqual(0, gfro_frags[0].fluid_parts.fluid_lambdas[3])
        tb_f: GFROFragment
        ob_f: OneBodyFragment
        tb_f, ob_f = from_frag.move2frag(to=fluid_1, orb=0, coeff=coeff, mutate=True)
        self.assertEqual(tb_f.fluid_parts.fluid_lambdas[0], old_lambda - coeff)
        fluid_total, og_total = tb_f.to_op() + ob_f.to_op(), obt2op(a) + tbt2op(b)
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
        st.lists(
            st.floats(-2, 2, allow_nan=False, allow_infinity=False),
            max_size=4,
            min_size=4,
        ).filter(lambda n: 0 not in n),
    )
    def test_dif_thetas_dif_dims_matrices(self, coeff, coeffs):
        tbt_op = (
            FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((0, 1), (0, 0), (1, 1), (1, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (0, 1), (0, 0)))
            + FermionOperator(coefficient=coeff, term=((1, 1), (1, 0), (1, 1), (1, 0)))
        )
        tbt_ten = get_n_body_tensor(tbt_op, n=2, m=4)
        np.testing.assert_array_equal(tbt_ten, tbt_ten.T)
        gfro_frag = gfro_decomp(tbt_ten)[0]
        one_lam = np.array([0, 0, 0, 0, coeff / 2, 0, 0, 0, 0, 0])
        f_1 = make_fr_tensor(lambdas=one_lam, thetas=gfro_frag.thetas, n=4)
        copy_lam = copy(gfro_frag.lambdas)
        copy_lam[4] = coeff / 2
        f_2 = make_fr_tensor(lambdas=copy_lam, thetas=gfro_frag.thetas, n=4)
        np.testing.assert_array_almost_equal(copy_lam + one_lam, gfro_frag.lambdas)
        np.testing.assert_array_almost_equal(f_1 + f_2, tbt_ten)
        fake_obt = np.diagflat(coeffs)
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
        # == expectation value check ==
        eigenvalues, eigenvectors = sp.linalg.eigh(
            qubit_operator_sparse(jordan_wigner(total_op)).toarray()
        )
        eigenvalues_f, eigenvectors_f = sp.linalg.eigh(
            qubit_operator_sparse(
                jordan_wigner(fake_tb_fluid.to_op() + fake_ob_fluid.to_op())
            ).toarray()
        )
        np.testing.assert_array_almost_equal(eigenvalues, eigenvalues_f)
        np.testing.assert_array_almost_equal(eigenvectors_f, eigenvectors)

    # == GFRO Tests ==
    def test_1b_and_2b_to_ops_artificial(self):
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
        gfro_frag = gfro_decomp(fake_hamiltonian)
        frag_details = gfro_frag[0]
        fluid_gfro: GFROFragment = frag_details.to_fluid()
        fluid_gfro_op = fluid_gfro.to_op()
        self.assertEqual(1, len(gfro_frag))
        self.assertEqual(frag_details.operators, fluid_gfro_op)

    def test_1b_2b_to_ops_h2(self):
        for frag in self.gfro_h2_frags:
            fluid_frag = frag.to_fluid()
            self.assertEqual(frag.operators, fluid_frag.to_op())

    def test_convert_one_body_to_f3(self):
        f3_frag = obt2fluid(self.H_obt)
        self.assertAlmostEqual(
            np.linalg.det(make_unitary(f3_frag.thetas, 4)), 1, places=7
        )
        f3_ops = fluid_ob2op(f3_frag)
        self.assertEqual(f3_frag.operators, self.H_ob_op)
        self.assertEqual(f3_ops, self.H_ob_op)
        np.testing.assert_array_almost_equal(
            self.H_obt, get_n_body_tensor(f3_ops, n=1, m=4)
        )

    def test_add_0_gfro(self):
        orb = random.randint(0, 3)
        a_coeff = 0
        ob_f3_frag = obt2fluid(self.H_obt)
        tb_f3_frags = [gfro2fluid(f) for f in self.gfro_h2_frags]
        from_frag = tb_f3_frags[0]
        lambdas = from_frag.lambdas
        thetas = from_frag.thetas
        coeff = from_frag.fluid_parts.fluid_lambdas[orb]
        m_from_frag_m_ob_f3_frag = from_frag.move2frag(
            orb=orb,
            to=ob_f3_frag,
            coeff=a_coeff,
            mutate=True,  # why is moving 0 changing it
        )
        m_from_frag: GFROFragment = m_from_frag_m_ob_f3_frag[0]
        m_ob_f3_frag: OneBodyFragment = m_from_frag_m_ob_f3_frag[1]
        self.assertAlmostEqual(
            m_from_frag.fluid_parts.fluid_lambdas[orb], coeff, places=8
        )
        a_orb, fluid_parts = m_ob_f3_frag.fluid_lambdas[0]
        np.testing.assert_array_equal(m_from_frag.lambdas, lambdas)
        np.testing.assert_array_equal(m_from_frag.thetas, thetas)
        self.assertEqual(orb, a_orb)
        self.assertAlmostEqual(fluid_parts.coeff, a_coeff, places=8)
        orig_op = self.H_ob_op + self.H_tb_op
        m_f3_tb_ten = m_from_frag.to_tensor()
        np.testing.assert_array_equal(self.H_tbt, m_f3_tb_ten)
        m_f3_ob_ten = m_ob_f3_frag.to_tensor()
        np.testing.assert_array_equal(self.H_obt, m_f3_ob_ten)
        m_f3_tbop = tbt2op(m_f3_tb_ten)
        m_f3_obop = obt2op(m_f3_ob_ten)
        rest_of_tb_op = reduce(
            lambda a, b: a + b,
            [f.operators for f in self.gfro_h2_frags[1:]],
            FermionOperator(),
        )
        fluid_frag = m_f3_obop + m_f3_tbop + rest_of_tb_op
        self.assertEqual(orig_op, fluid_frag)

    def test_sub_coeff_exactly_gfro(self):
        orb = random.randint(0, 3)
        frag = random.randint(0, len(self.gfro_h2_frags) - 1)
        ob_f3_frag = obt2fluid(self.H_obt)
        tb_f3_frags = [gfro2fluid(f) for f in self.gfro_h2_frags]
        from_frag = tb_f3_frags[frag]
        lambdas = from_frag.lambdas
        thetas = from_frag.thetas
        coeff = from_frag.fluid_parts.fluid_lambdas[orb]
        m_from_frag_m_ob_f3_frag = from_frag.move2frag(
            orb=orb,
            to=ob_f3_frag,
            coeff=coeff,
            mutate=True,  # why is moving 0 changing it
        )
        m_from_frag: GFROFragment = m_from_frag_m_ob_f3_frag[0]
        m_ob_f3_frag: OneBodyFragment = m_from_frag_m_ob_f3_frag[1]
        self.assertEqual(m_from_frag.fluid_parts.fluid_lambdas[orb], 0)
        a_orb, fluid_parts = m_ob_f3_frag.fluid_lambdas[0]
        np.testing.assert_array_equal(m_from_frag.lambdas, lambdas)
        np.testing.assert_array_equal(m_from_frag.thetas, thetas)
        self.assertEqual(orb, a_orb)
        self.assertEqual(fluid_parts.coeff, coeff)
        orig_op = self.H_ob_op + self.H_tb_op
        m_f3_tb_ten = m_from_frag.to_tensor()
        m_f3_ob_ten = m_ob_f3_frag.to_tensor()
        m_f3_tbop = tbt2op(m_f3_tb_ten)
        m_f3_obop = obt2op(m_f3_ob_ten)
        rest_of_tb_op = reduce(
            lambda a, b: a + b,
            [
                f.operators if i != orb else FermionOperator()
                for i, f in enumerate(self.gfro_h2_frags)
            ],
            FermionOperator(),
        )
        fluid_frag = m_f3_obop + m_f3_tbop + rest_of_tb_op
        self.assertEqual(orig_op, fluid_frag)

    def test_add_b_eq_a_coeff_gfro(self):
        pass

    def test_add_b_less_a_coeff_gfro(self):
        pass

    def test_add_b_more_a_coeff_gfro(self):
        pass

    def test_mutate_each_frag_gfro(self):
        pass

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
