from copy import deepcopy
from functools import partial

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from d_types.config_types import MConfig
from d_types.fragment_types import OneBodyFragment, GFROFragment
from d_types.hamiltonian import FragmentedHamiltonian
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


def find_fluid_coeffs(
    self: FragmentedHamiltonian, frag_i: int, starting_E: float, iters: int
):
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
    bounds = [(0, c) if c > 0 else (c, 0) for c in obp]
    x0 = np.zeros(n)

    def cost(x0_0):
        obf_copy = deepcopy(obf)
        tbf_copy = deepcopy(tbf)
        for i in range(n):
            tbf_copy.move2frag(to=obf_copy, coeff=x0_0[i], orb=i, mutate=True)
        obp_E = self._diagonalize_operator_with_ss_proj(
            self.constant + obf_copy.to_op()
        )
        tbp_E = self._diagonalize_operator_with_ss_proj(tbf_copy.operators)
        return starting_E - (obp_E + tbp_E)

    greedy_coeffs: OptimizeResult = minimize(
        cost,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "maxiter": iters,
            # "disp": False
        },
    )
    return greedy_coeffs


def greedy_fluid_optimize(self: FragmentedHamiltonian, iters: int, debug: bool = False):
    """
    Mimimizes each LR fragment at once.

    Args:
        self:
        iters:

    Returns:

    """
    n = self.one_body.lambdas.shape[0]
    if self.partitioned and not self.fluid:
        self.fluid = True
        for i in range(len(self.two_body)):
            if debug:
                print(f"Optimizing fragment: {i}")
            obp_E = self._diagonalize_operator_with_ss_proj(
                self.constant + self.one_body.to_op()
            )
            og_tbp_E = self._filter_frag_energy(self.two_body[i])
            starting_E = obp_E + og_tbp_E
            self.two_body[i].to_fluid()
            frag_i_coeffs = find_fluid_coeffs(self, i, starting_E, iters)
            for c in range(n):
                self.two_body[i].move2frag(
                    to=self.one_body, coeff=frag_i_coeffs.x[c], orb=c, mutate=True
                )
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
