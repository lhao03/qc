import time
import unittest


from d_types.config_types import PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.molecules import h2o_settings
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
    ham.partition(strategy=ps, bond_length=bond_length, save=True)
    end_p = time.time()
    print(f"finished partitioning: {bond_length} in {end_p - start_p}")


class ParaTest(unittest.TestCase):
    def test_para_part(self):
        num_cpus = 4
        ray.init(num_cpus=num_cpus)
        m_config = h2o_settings
        futures = [
            partition_frags.remote(b, m_config, PartitionStrategy.GFRO)
            for b in m_config.xpoints
        ]
        res = ray.get(futures)

    def test_water(self):
        start_p = time.time()
        m_config = h2o_settings
        bond_length = h2o_settings.stable_bond_length
        const, obt, tbt = get_tensors(m_config, bond_length)
        ham = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
        )
        ham.partition(
            strategy=PartitionStrategy.GFRO, bond_length=bond_length, save=True
        )
        end_p = time.time()
        print(f"finished partitioning: {bond_length} in {end_p - start_p}")
