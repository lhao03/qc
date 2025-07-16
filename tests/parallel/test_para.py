import time
import unittest

import ipyparallel as ipp

from d_types.config_types import PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.molecules import h4_settings, h4_sq_settings
from tests.utils.sim_tensor import get_tensors
import ray


@ray.remote
def partition_frags(bond_length, m_config, ps):
    start_p = time.time()
    const, obt, tbt = get_tensors(m_config, bond_length)
    ham = FragmentedHamiltonian(
        m_config=m_config,
        constant=const,
        one_body=obt,
        two_body=tbt,
        partitioned=False,
        fluid=False,
    )
    ham.partition(strategy=ps, bond_length=bond_length, save=True, load_prev=False)
    end_p = time.time()
    print(f"per iter time: {end_p - start_p}")


@ray.remote
def get_num(i):
    return i


class ParaTest(unittest.TestCase):
    def test_ipypara(self, bond_lengths: list):
        n = 4
        cluster = ipp.Cluster(n=n)
        c = cluster.start_and_connect_sync()
        lview = c.load_balanced_view()
        lview.block = True
        frags = lview.map(partition_frags, bond_lengths)
        cluster.stop_cluster()

    def test_para_part(self):
        num_cpus = 7
        ray.init(num_cpus=num_cpus)
        m_config = h4_settings
        futures = [
            partition_frags.remote(b, m_config, PartitionStrategy.GFRO)
            for b in m_config.xpoints
        ]
        res = ray.get(futures)

    def test_hh_hh(self):
        num_cpus = 4
        ray.init(num_cpus=num_cpus)
        m_config = h4_sq_settings
        futures = [
            partition_frags.remote(b, m_config, PartitionStrategy.GFRO)
            for b in m_config.xpoints
        ]
        res = ray.get(futures)
