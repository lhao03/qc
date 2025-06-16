import random
import unittest
from typing import List, Tuple

import numpy as np
from openfermion import (
    count_qubits,
    FermionOperator,
)

from d_types.fragment_types import GFROFragment, OneBodyFragment
from min_part.f_3_ops import (
    obp_of_tbp_2t,
    obt2fluid,
    collect_ob2op,
    gfro2fluid,
)
from min_part.gfro_decomp import gfro_decomp, make_unitary, make_fr_tensor_from_u
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.lr_decomp import lr_decomp
from min_part.molecules import mol_h2
from min_part.tensor import get_n_body_tensor
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class FluidFragmentTest(unittest.TestCase):
    bond_length = 0.80
    mol = mol_h2(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
    H_ob_op = obt2op(H_obt)
    H_tb_op = tbt2op(H_tbt)

    def setUp(self):
        self.gfro_h2_frags = gfro_decomp(self.H_tbt)
        self.lr_h2_frags = lr_decomp(self.H_tbt)

    def test_get_one_body_parts(self):
        n = 5
        m = (n * (n + 1)) // 2
        fake_h2 = np.random.rand(m)
        diags = [fake_h2[0], fake_h2[5], fake_h2[9], fake_h2[12], fake_h2[14]]
        gfro_frag = GFROFragment(
            lambdas=np.array(fake_h2), thetas=[], operators=FermionOperator()
        )
        self.assertEqual(diags, gfro_frag.get_ob_lambdas())

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
            fluid_ops = obt2op(obp_of_tbp_2t(frag.get_ob_lambdas(), thetas=frag.thetas))
            static_ops = frag.to_op()
            self.assertEqual(frag.operators, fluid_ops + static_ops)

    def test_convert_one_body_to_f3(self):
        f3_frag = obt2fluid(self.H_obt)
        self.assertAlmostEqual(
            np.linalg.det(make_unitary(f3_frag.thetas, 4)), 1, places=7
        )
        f3_ops = collect_ob2op(
            lambdas=f3_frag.fluid_lambdas,
            thetas=f3_frag.thetas,
            diag_thetas=f3_frag.diag_thetas,
        )
        self.assertEqual(f3_frag.operators, self.H_ob_op)
        self.assertEqual(f3_ops, self.H_ob_op)
        np.testing.assert_array_almost_equal(
            self.H_obt, get_n_body_tensor(f3_ops, n=1, m=4)
        )

    def test_convert_gfro_2b_to_f3_fake(self):
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
        fake_hamiltonian_operator = tbt2op(fake_hamiltonian)
        gfro_frags = gfro_decomp(fake_hamiltonian)
        frag_details = gfro_frags[0]
        gfro_fluid = frag_details.to_fluid()
        fluid_frags = gfro_fluid.fluid_parts.fluid_lambdas
        static_frags = gfro_fluid.fluid_parts.static_lambdas
        self.assertEqual(
            fake_hamiltonian_operator,
            obp_of_tbp_2t(fluid_frags, gfro_fluid.thetas) + gfro_fluid.to_op(),
        )
        self.assertEqual(
            fake_hamiltonian_operator,
            obp_of_tbp_2t(fluid_frags, gfro_fluid.thetas)
            + gfro_fluid.to_op(),  # TODO test static frags
        )

    def test_convert_gfro_2b_to_f3_h2(self):
        fluid_h2_frags: List[GFROFragment] = [f.to_fluid() for f in self.gfro_h2_frags]
        for fff in fluid_h2_frags:
            self.assertEqual(
                fff.operators, obp_of_tbp_2t(fff.fluid_parts, fff.thetas) + fff.to_op()
            )

    def test_add_b_eq_a_coeff_gfro(self):
        ob_f3_frag = obt2fluid(self.H_obt)
        tb_f3_frags = [gfro2fluid(f) for f in self.gfro_h2_frags]
        from_frag = tb_f3_frags[0]
        coeff = from_frag.fluid_parts.fluid_lambdas[0]
        m_from_frag, m_ob_f3_frag = from_frag.move2frag(
            orb=1, to=ob_f3_frag, coeff=coeff, mutate=True
        )
        orig_op = self.H_ob_op + self.H_tb_op
        self.assertEqual(orig_op, m_from_frag.to_op())

    def test_add_b_less_a_coeff_gfro(self):
        pass

    def test_add_b_more_a_coeff_gfro(self):
        pass

    def test_mutate_each_frag_gfro(self):
        pass

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
