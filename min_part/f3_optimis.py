from copy import deepcopy
from functools import partial
from typing import List

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from d_types.config_types import MConfig
from d_types.fragment_types import OneBodyFragment, GFROFragment
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_opers import move_ob_to_ob
from min_part.operators import (
    get_particle_number,
    get_projected_spin,
    get_total_spin,
    subspace_projection_operator,
)


def subspace_operators(m_config: MConfig):
    number_operator = partial(get_particle_number, e=m_config.num_spin_orbs)
    sz = partial(get_projected_spin, p=m_config.num_spin_orbs // 2)
    s2 = partial(get_total_spin, p=m_config.num_spin_orbs // 2)
    return number_operator, sz, s2


def cost_part_E_sub_curr_E(
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
        method="L-BFGS-B",
        options={"maxiter": iters, "disp": False},
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
            frag_i_coeffs = cost_part_E_sub_curr_E(self, i, starting_E, iters)
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


# == Simple Test Cases ==
def greedy_E_optimize(
    ob: OneBodyFragment, frags: List[OneBodyFragment], iters: int, debug: bool = False
):
    def min_eig(fo):
        return min(np.linalg.eigh(subspace_projection_operator(fo, n, 2).toarray())[0])

    n = frags[0].lambdas.shape[0]
    for i in range(len(frags)):
        if debug:
            print(f"Optimizing fragment: {i}")
        obp_E = min_eig(ob.to_op())
        og_tbp_E = min_eig(frags[i].to_op())
        starting_E = obp_E + og_tbp_E
        x0 = np.random.random(n)

        def cost(x0_0):
            obf_copy = deepcopy(ob)
            frag_copy = deepcopy(frags[i])
            for j in range(n):
                move_ob_to_ob(from_ob=frag_copy, to_ob=obf_copy, coeff=x0_0[j], orb=j)
            new_o_E = min_eig(obf_copy.to_op())
            new_f_E = min_eig(frag_copy.to_op())
            new_E = new_o_E + new_f_E
            return starting_E - new_E

        coeffs = minimize(
            cost,
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": iters, "disp": False},
        )

        for c in range(n):
            move_ob_to_ob(from_ob=frags[i], to_ob=ob, coeff=coeffs.x[c], orb=c)
    print("Complete")
