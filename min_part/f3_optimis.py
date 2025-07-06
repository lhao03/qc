from copy import deepcopy
from functools import partial
from typing import List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from d_types.config_types import MConfig
from d_types.fragment_types import OneBodyFragment
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


def afao_fluid_optimize(self: FragmentedHamiltonian, iters: int, debug: bool = False):
    n = self.one_body.lambdas.shape[0]
    if self.partitioned and not self.fluid:
        self.fluid = True
        for f in self.two_body:
            if f.fluid_parts is None:
                f.to_fluid()
        obp_E = self._diagonalize_operator_with_ss_proj(self.one_body.to_op())
        tbp_E = sum(
            [self._diagonalize_operator_with_ss_proj(f.to_op()) for f in self.two_body]
        )
        starting_E = obp_E + tbp_E

        def cost(x0_0):
            obf_copy = deepcopy(self.one_body)
            tbf_copy = [deepcopy(f) for f in self.two_body]
            for j, f in enumerate(tbf_copy):
                coeffs = x0_0[j * n : j * n + n]
                f.bulkmove2frag(to=obf_copy, coeffs=coeffs)
            obp_E = self._diagonalize_operator_with_ss_proj(obf_copy.to_op())
            tbp_E = sum(
                [self._diagonalize_operator_with_ss_proj(f.to_op()) for f in tbf_copy]
            )
            new_E = obp_E + tbp_E
            return starting_E - new_E

        x0 = np.random.rand(n * len(self.two_body)) * 1000
        frag_i_coeffs: OptimizeResult = minimize(
            cost,
            x0=x0,
            method="CG",
            options={"maxiter": iters, "disp": False},
        )
        for i in range(len(self.two_body)):
            coeffs = frag_i_coeffs.x[i * n : i * n + n]
            self.two_body[i].bulkmove2frag(to=self.one_body, coeffs=coeffs)
    return self


def ofat_fluid_optimize(self: FragmentedHamiltonian, iters: int, debug: bool = False):
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
            obp_E = self._diagonalize_operator_with_ss_proj(self.one_body.to_op())
            og_tbp_E = self._diagonalize_operator_with_ss_proj(self.two_body[i].to_op())
            starting_E = obp_E + og_tbp_E
            if self.two_body[i].fluid_parts is None:
                self.two_body[i].to_fluid()

            def cost(x0_0):
                obf_copy = deepcopy(self.one_body)
                tbf_copy = deepcopy(self.two_body[i])
                tbf_copy.bulkmove2frag(to=obf_copy, coeffs=x0_0)
                obp_E = self._diagonalize_operator_with_ss_proj(obf_copy.to_op())
                tbp_E = self._diagonalize_operator_with_ss_proj(tbf_copy.to_op())
                new_E = obp_E + tbp_E
                return starting_E - new_E

            x0 = np.random.rand(n)
            frag_i_coeffs: OptimizeResult = minimize(
                cost,
                x0=x0,
                method="CG",
                options={"maxiter": iters, "disp": False},
            )
            self.two_body[i].bulkmove2frag(to=self.one_body, coeffs=frag_i_coeffs.x)
    return self


def find_region_using_bisection(
    self: FragmentedHamiltonian,
    j: int,
    orb: int,
    lower: float = -100,
    upper: float = 100,
    second_try: bool = False,
):
    starting_E = self.get_expectation_value()

    def get_diff(x0):
        ham_copy = deepcopy(self)
        ham_copy.two_body[j].move2frag(
            ham_copy.one_body, orb=orb, coeff=x0, mutate=True
        )
        new_E = ham_copy.get_expectation_value()
        return starting_E - new_E

    f_a = get_diff(lower)
    f_b = get_diff(upper)
    if f_a < 0 and f_b > 0:
        return lower, upper
    elif f_a < 0 and f_b < 0:
        i = 0
        while f_b <= 0:
            upper = 2**i
            f_b = get_diff(upper)
            if i == 12:
                if second_try:
                    raise UserWarning("Couldn't find region")
                lower = np.random.randint(-100, 100)
                upper = lower + 1
                return find_region_using_bisection(
                    self, j, orb, lower=lower, upper=upper, second_try=True
                )
    elif f_a > 0 and f_b > 0:
        i = 1
        while f_a > 0:
            lower = -(2**i)
            f_a = get_diff(lower)
            i += 1
            if i == 13:
                if second_try:
                    raise UserWarning("Couldn't find region")
                lower = np.random.randint(-100, 100)
                upper = lower + 1
                return find_region_using_bisection(
                    self, j, orb, lower=lower, upper=upper, second_try=True
                )
    elif f_a > 0 and f_b < 0:
        if second_try:
            raise UserWarning("Couldn't find region")
        lower = np.random.randint(-100, 100)
        upper = lower + 1
        return find_region_using_bisection(
            self, j, orb, lower=lower, upper=upper, second_try=True
        )
    raise UserWarning("Shouldn't get here")


def greedy_coeff_optimize(
    self: FragmentedHamiltonian, iters: int, threshold: float, debug: bool = False
):
    self_copy = deepcopy(self)
    starting_E = self_copy.get_expectation_value()
    tries = 0
    diff = 100
    n = len(self_copy.one_body.lambdas)
    while tries > iters or diff >= threshold:
        diffs = []
        for j, f in enumerate(self_copy.two_body):
            if f.fluid_parts is None:
                f.to_fluid()
            for i in range(n):

                def cost(x0):
                    ham_copy = deepcopy(self_copy)
                    ham_copy.two_body[j].move2frag(
                        ham_copy.one_body, orb=i, coeff=x0[0], mutate=True
                    )
                    new_E = ham_copy.get_expectation_value()
                    return starting_E - new_E

                x0 = f.fluid_parts.fluid_lambdas[i]
                c: OptimizeResult = minimize(
                    cost,
                    x0=x0,
                    method="CG",
                    options={"maxiter": iters, "disp": False},
                )
                diffs.append((c.x[0], cost(c.x)))
        best_diff = min(diffs, key=lambda d: d[1])
        best_ind = best_diff[0]
        frag = best_ind // n
        orb = best_ind - (frag * n) - 1
        self_copy.two_body[frag].move2frag(
            to=self_copy.one_body, orb=orb, coeff=best_diff[1], mutate=True
        )
        diff = starting_E - self_copy.get_expectation_value()
        starting_E = self_copy.get_expectation_value()


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
