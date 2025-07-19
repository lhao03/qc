import time


from d_types.hamiltonian import FragmentedHamiltonian
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
