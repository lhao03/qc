from copy import deepcopy
from functools import partial

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from d_types.config_types import MConfig
from d_types.fragment_types import OneBodyFragment, GFROFragment
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_opers import obt2fluid
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


def find_fluid_coeffs_gfro(self: FragmentedHamiltonian, frag_i: int, iters: int):
    """
    Calculate the expectation value for the one electron part and one two electron GFRO fragment.
    Args:


    Returns:

    """
    obf: OneBodyFragment = self.one_body
    tbf: GFROFragment = self.two_body[frag_i]
    obp = tbf.fluid_parts.fluid_lambdas
    n = obp.shape[0]
    if not n == obf.lambdas.shape[0]:
        raise UserWarning(
            "Expected the same number of one body parts as dimension of one-electron part"
        )
    x0 = np.random.uniform(low=min(obp), high=max(obp), size=obp.size)

    def cost(x0_0):
        obf_copy = deepcopy(obf)
        tbf_copy = deepcopy(tbf)
        for i in range(n):
            tbf_copy.move2frag(to=obf_copy, coeff=x0_0[i], orb=i, mutate=True)
        obp_E = self._diagonalize_operator(self.constant + obf_copy.to_op())
        tbp_E = self._diagonalize_operator(tbf_copy.to_op())
        return obp_E + tbp_E

    bounds = [(0, c) if c > 0 else (c, 0) for c in obp]
    greedy_coeffs: OptimizeResult = minimize(
        cost,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": iters, "disp": False},
    )
    return greedy_coeffs


def greedy_fluid_gfro_optimize(
    self: FragmentedHamiltonian, iters: int, debug: bool = False
):
    """
    Mimimizes each GFRO fragment at once, starting with the largest.

    Args:
        self:
        iters:

    Returns:

    """
    if self.partitioned and not self.fluid:
        self.fluid = True
        self.one_body = obt2fluid(self.one_body)
        for i in range(len(self.two_body)):
            if debug:
                print(f"Optimizing fragment: {i}")
            self.two_body[i].to_fluid()
            frag_i_coeffs = find_fluid_coeffs_gfro(self, i, iters)
            self.two_body[i].bulkmove2frag(self.one_body, frag_i_coeffs.x)
    return self


def lp_fluid_gfro_optimize(
    self: FragmentedHamiltonian,
    iters: int,
):
    pass


def lr_cost_func(self: FragmentedHamiltonian):
    pass


def greedy_fluid_lr_optimize(
    self: FragmentedHamiltonian,
    iters: int,
):
    pass


def lp_fluid_lr_optimize(
    self: FragmentedHamiltonian,
    iters: int,
):
    pass
