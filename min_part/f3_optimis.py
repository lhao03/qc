from functools import partial

from d_types.config_types import MConfig
from min_part.operators import get_particle_number, get_projected_spin, get_total_spin


def subspace_operators(m_config: MConfig):
    number_operator = partial(get_particle_number, e=m_config.num_spin_orbs)
    sz = partial(get_projected_spin, p=m_config.num_spin_orbs // 2)
    s2 = partial(get_total_spin, p=m_config.num_spin_orbs // 2)
    return number_operator, sz, s2


def optimization_checks():
    pass


def sanity_checks():
    pass
