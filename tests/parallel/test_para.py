import unittest

import ipyparallel as ipp
from d_types.config_types import PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.molecules import h4_settings
from tests.utils.sim_tensor import get_tensors


class CuPyTest(unittest.TestCase):
    def test_optimize_fragments(self, m_config=h4_settings, bond_length=0.8):
        const, obt, tbt = get_tensors(m_config, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
        )
        ham.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        print(ham.get_expectation_value())

    def test_ipypara(self):
        n = 4
        cluster = ipp.Cluster(n=n)
        c = cluster.start_and_connect_sync()
        dview = c[:]
        dview.block = True
        dview.apply(lambda: "Hello, World")
        print(dview)
