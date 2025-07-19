import unittest
from copy import deepcopy

import numpy as np

from d_types.config_types import PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.molecules import h2_settings, h2o_settings
from min_part.tensor import tbt2op, obt2op, spin2spac, spac2spin
from tests.utils.sim_tensor import get_tensors


class TestSpinSpac(unittest.TestCase):
    def test_spin2spac_tensor(self):
        _, _, tbt = get_tensors(h2_settings, 0.8)
        spinten = tbt
        spacten = spin2spac(spinten)
        np.testing.assert_array_equal(spinten, spac2spin(spacten))

    def test_lr_spin2spac(self):
        pass

    def test_lr_spac2spin(self):
        pass

    def test_gfro_spin2spac_h2(self):
        m_config = h2_settings
        const, obt, tbt = get_tensors(m_config, bond_length=0.8, load=True)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
        )
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=0.8)
        old_frags = deepcopy(ham.two_body)
        one_ob = deepcopy(ham.one_body)
        ham.spin2spac()
        ham.partition(
            strategy=PartitionStrategy.GFRO,
            bond_length=0.8,
            load_prev=False,
            save=False,
        )
        ham.spac2spin()
        self.assertEqual(
            sum([f.to_op() for f in old_frags]), sum([f.to_op() for f in ham.two_body])
        )
        self.assertEqual(ham.one_body.to_op(), one_ob.to_op())

    def test_gfro_spin2spac_h2o(self):
        # water
        m_config = h2o_settings
        const, obt, tbt = get_tensors(m_config, bond_length=0.95, load=True)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
        )
        ham.spin2spac()
        ham.partition(
            strategy=PartitionStrategy.GFRO,
            bond_length=0.8,
            load_prev=False,
            save=False,
        )
        ham.spac2spin()
        self.assertEqual(
            const + obt2op(obt) + tbt2op(tbt),
            ham.constant
            + ham.one_body.to_op()
            + sum([f.to_op() for f in ham.two_body]),
        )

    def test_gfro_spac2spin(self):
        pass
