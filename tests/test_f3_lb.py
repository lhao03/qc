import unittest
from typing import Tuple

import numpy as np
from openfermion import count_qubits

from d_types.config_types import MConfig
from d_types.fragment_types import Subspace, PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_optimis import subspace_operators
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import h2_settings
from tests.utils.sim_tensor import get_chem_tensors


def get_tensors(
    m_config: MConfig, bond_length: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    mol = m_config.mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


class F3Test(unittest.TestCase):
    # == saving nums
    no_partitioning = []
    lr = []
    lr_f3 = []
    gfro = []
    gfro_f3 = []

    def test_partition(self, bond_length: float, m_config: MConfig):
        number_operator, sz, s2 = subspace_operators(m_config)
        const, obt, tbt = get_tensors(m_config, bond_length)
        reference = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        lr = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        E = reference.get_expectation_value()
        gfro_frags = gfro.partition(
            strategy=PartitionStrategy.GFRO, bond_length=bond_length
        )
        lr_frags = lr.partition(strategy=PartitionStrategy.LR, bond_length=bond_length)
        self.assertEqual(
            sum([f.operators for f in gfro_frags]), sum([f.operators for f in lr_frags])
        )
        E_gfro = gfro.get_expectation_value()
        E_lr = lr.get_expectation_value()
        self.assertTrue(E >= E_gfro)
        self.assertTrue(E >= E_lr)

    def test_make_lb(self):
        self.test_partition(bond_length=1, m_config=h2_settings)

    def test_ask_simple(self, bond_length: float = 1, m_config=h2_settings):
        number_operator, sz, s2 = subspace_operators(m_config)
        const, obt, tbt = get_tensors(m_config, bond_length)
        self.assertTrue(isinstance(const, float))
        self.assertEqual(obt.shape, (m_config.num_spin_orbs, m_config.num_spin_orbs))
        self.assertEqual(
            tbt.shape,
            (
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
            ),
        )
        reference = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        E = reference.get_expectation_value()
        self.assertTrue(-2 <= E <= 0)
        saved_location = reference.save()
        loaded_reference = FragmentedHamiltonian.load(saved_location)
        self.assertEqual(reference, loaded_reference)
