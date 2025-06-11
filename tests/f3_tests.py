import random
import unittest

import numpy as np
from openfermion import (
    count_qubits,
)

from min_part.f_3_ops import (
    get_one_body_parts,
    remove_one_body_parts,
    oneb2op,
    twob2op,
    obt2fluid,
)
from min_part.gfro_decomp import make_fr_tensor_from_u, gfro_decomp, make_unitary
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import mol_h2
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
        fluid_ops = oneb2op(get_one_body_parts(lambdas=frag_details.lambdas), thetas=frag_details.thetas)
        static_ops = twob2op(remove_one_body_parts(lambdas=frag_details.lambdas), thetas=frag_details.thetas)
        self.assertEqual(1, len(gfro_frag))
        self.assertEqual(frag_details.operators, fluid_ops + static_ops)

    def test_1b_2b_to_ops_h2(self):
        h2_frags = gfro_decomp(self.H_tbt)
        for frag in h2_frags:
            fluid_ops = oneb2op(get_one_body_parts(frag.lambdas), frag.thetas)
            static_ops = twob2op(remove_one_body_parts(frag.lambdas), thetas=frag.thetas)
            self.assertEqual(frag.operators, fluid_ops + static_ops)

    def test_convert_one_body_to_f3(self):
        f3_frag = obt2fluid(self.H_obt)
        self.assertAlmostEqual(np.linalg.det(make_unitary(f3_frag.thetas, 4)), 1, places=7)

    def test_convert_two_body_to_f3(self):
        pass

    def test_move_from_2b_2_1b(self):
        pass

    def test_move_from_2b_2_1b_multiple(self):
        pass

    def test_rediag_1b(self):
        pass
