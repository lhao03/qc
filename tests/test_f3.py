import random
import unittest

import numpy as np
from openfermion import (
    count_qubits,
)

from d_types.fragment_types import FluidCoeff
from min_part.f_3_ops import (
    get_one_body_parts,
    remove_one_body_parts,
    oneb2op,
    twob2op,
    obt2fluid,
    extract_thetas,
    obf3to_op,
    fragment2fluid,
)
from min_part.gfro_decomp import gfro_decomp, make_unitary, make_fr_tensor_from_u
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
from min_part.tensor import get_n_body_tensor
from min_part.tensor_utils import get_chem_tensors, obt2op, tbt2op


class FluidFragmentTest(unittest.TestCase):
    def setUp(self):
        bond_length = 0.80
        self.mol = mol_h2(bond_length)
        H, num_elecs = obtain_OF_hamiltonian(self.mol)
        self.n_qubits = count_qubits(H)
        self.H_const, self.H_obt, self.H_tbt = get_chem_tensors(H=H, N=self.n_qubits)
        self.H_ob_op = obt2op(self.H_obt)
        self.H_tb_op = tbt2op(self.H_tbt)
        self.H_ele = self.H_const + self.H_ob_op + self.H_tb_op
        self.gfro_h2_frags = gfro_decomp(self.H_tbt)

    def test_get_one_body_parts(self):
        n = 5
        m = (n * (n + 1)) // 2
        fake_h2 = np.random.rand(m)
        diags = [fake_h2[0], fake_h2[5], fake_h2[9], fake_h2[12], fake_h2[14]]
        self.assertEqual(diags, get_one_body_parts(np.array(fake_h2)))

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
        fluid_ops = oneb2op(
            FluidCoeff(
                coeff=get_one_body_parts(lambdas=frag_details.lambdas),
                thetas=frag_details.thetas,
            )
        )
        static_ops = twob2op(
            remove_one_body_parts(lambdas=frag_details.lambdas),
            thetas=frag_details.thetas,
        )
        self.assertEqual(1, len(gfro_frag))
        self.assertEqual(frag_details.operators, fluid_ops + static_ops)

    def test_1b_2b_to_ops_h2(self):
        for frag in self.gfro_h2_frags:
            fluid_ops = oneb2op(
                FluidCoeff(coeff=get_one_body_parts(frag.lambdas), thetas=frag.thetas)
            )
            static_ops = twob2op(
                remove_one_body_parts(frag.lambdas), thetas=frag.thetas
            )
            self.assertEqual(frag.operators, fluid_ops + static_ops)

    def test_extract_thetas(self):
        fake_u = np.array(
            [
                [0.70710029, 0.00303002, 0.70710028, 0.00303002],
                [-0.00303002, 0.70710029, -0.00303002, 0.70710028],
                [-0.70710493, -0.00161596, 0.70710494, 0.00161596],
                [0.00161597, -0.70710493, -0.00161596, 0.70710494],
            ]
        )
        thetas = extract_thetas(fake_u)
        made_u = make_unitary(thetas, 4)
        self.assertTrue(np.allclose(fake_u, made_u))

    def test_convert_one_body_to_f3(self):
        f3_frag = obt2fluid(self.H_obt)
        self.assertAlmostEqual(
            np.linalg.det(make_unitary(f3_frag.thetas, 4)), 1, places=7
        )
        f3_ops = obf3to_op(lambdas=f3_frag.static_frags, thetas=f3_frag.thetas)
        self.assertEqual(f3_frag.operators, self.H_ob_op)
        self.assertEqual(f3_ops, self.H_ob_op)
        self.assertTrue(np.allclose(self.H_obt, get_n_body_tensor(f3_ops, n=1, m=4)))

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
        gfro_fluid = fragment2fluid(frag_details)
        fluid_frags = gfro_fluid.fluid_frags
        static_frags = gfro_fluid.static_frags
        self.assertEqual(
            fake_hamiltonian_operator,
            oneb2op(fluid_frags[0]) + twob2op(static_frags, gfro_fluid.thetas),
        )

    def test_convert_gfro_2b_to_f3_h2(self):
        fluid_h2_frags = [fragment2fluid(f) for f in self.gfro_h2_frags]
        for fff in fluid_h2_frags:
            self.assertEqual(
                fff.operators,
                oneb2op(fff.fluid_frags[0]) + twob2op(fff.static_frags, fff.thetas),
            )

    def test_convert_lr_2b_to_f3(self):
        pass

    def test_move_from_2b_2_1b(self):
        pass

    def test_move_from_2b_2_1b_multiple(self):
        pass

    def test_rediag_1b(self):
        pass
