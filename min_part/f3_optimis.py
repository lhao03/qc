from copy import deepcopy
from functools import partial, reduce
from typing import List, Callable

import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize, OptimizeResult
import cvxpy as cp

from d_types.config_types import MConfig
from d_types.fragment_types import OneBodyFragment, FermionicFragment, ContractPattern
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_opers import move_ob_to_ob, make_unitary_jl
from min_part.julia_ops import jl_make_u
from min_part.operators import (
    get_particle_number,
    get_projected_spin,
    get_total_spin,
)


def subspace_operators(m_config: MConfig):
    number_operator = partial(get_particle_number, e=m_config.num_spin_orbs)
    sz = partial(get_projected_spin, p=m_config.num_spin_orbs // 2)
    s2 = partial(get_total_spin, p=m_config.num_spin_orbs // 2)
    return number_operator, sz, s2


def get_bounds(frags: List[FermionicFragment]):
    bounds = []
    for f in frags:
        for c in f.fluid_parts.fluid_lambdas:
            if c > 0:
                bounds.append((0, c))
            else:
                bounds.append((c, 0))
    return bounds


def afao_fluid_optimize(self: FragmentedHamiltonian, iters: int, debug: bool = False):
    n = self.one_body.lambdas.shape[0]
    if self.partitioned and not self.fluid:
        self.fluid = True
        for f in self.two_body:
            if f.fluid_parts is None:
                f.to_fluid()

        starting_E = self.get_expectation_value()

        def cost(x0_0):
            self_copy = deepcopy(self)
            for j, f in enumerate(self_copy.two_body):
                coeffs = x0_0[j * n : j * n + n]
                f.bulkmove2frag(to=self_copy.one_body, coeffs=coeffs)
            new_E = self_copy.get_expectation_value()
            return starting_E - new_E

        bounds = get_bounds(self.two_body)
        x0 = np.zeros(n * len(self.two_body))
        frag_i_coeffs: OptimizeResult = minimize(
            cost,
            x0=np.array(x0),
            bounds=bounds,
            method="Nelder-Mead",
            options={"maxiter": iters, "disp": False},
        )
        for i in range(len(self.two_body)):
            coeffs = frag_i_coeffs.x[i * n : i * n + n]
            self.two_body[i].bulkmove2frag(to=self.one_body, coeffs=coeffs)
    return self


def ofat_fluid_optimize(self: FragmentedHamiltonian, iters: int, debug: bool = False):
    """
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


def convex_optimization(self: FragmentedHamiltonian, min_eig: Callable):
    n = self.one_body.lambdas.shape[0]
    num_coeffs = []
    for f in self.two_body:
        f.to_fluid()
        num_coeffs = np.append(num_coeffs, f.fluid_parts.fluid_lambdas)
    coeffs = [cp.Variable() for _ in range(n * len(self.two_body))]
    geq_con = [
        c >= 0 if (num_coeffs[i] > 0) else c >= num_coeffs[i]
        for i, c in enumerate(coeffs)
    ]
    leq_con = [
        c <= num_coeffs[i] if (num_coeffs[i] > 0) else c <= 0
        for i, c in enumerate(coeffs)
    ]
    constraints = geq_con + leq_con
    ob_t = self.one_body.to_tensor()
    unitaries = [make_unitary_jl(n, f) for f in self.two_body]
    frag_tensors = [
        contract(
            ContractPattern.GFRO.value,
            f.fluid_parts.fluid_lambdas,
            unitaries[i],
            unitaries[i],
        )
        for i, f in enumerate(self.two_body)
    ]
    print(f"optimal eigenvalue sum: {np.linalg.eigh(ob_t + sum(frag_tensors))[0]}")
    print(
        f"starting eigenvalue sum: {min_eig(ob_t) + sum([min_eig(f) for f in frag_tensors])}"
    )
    fluid_lambdas = [
        cp.diag(cp.vstack(coeffs[j * n : (j * n) + n]))
        for j in range(len(self.two_body))
    ]
    fluid_tensors = [
        np.linalg.inv(unitaries[i]) @ fluid_lambdas[i] @ unitaries[i]
        for i in range(len(self.two_body))
    ]

    new_obt = ob_t
    for i, fluid_tensor in enumerate(fluid_tensors):
        new_obt = new_obt + fluid_tensor
        frag_tensors[i] = frag_tensors[i] - fluid_tensor

    objective = cp.Maximize(
        cp.lambda_min(new_obt) + cp.sum([cp.lambda_min(m) for m in frag_tensors])
    )
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)
    print("optimal value", problem.value)
    optimal_coeffs: List[float] = [c.value for c in coeffs]
    for i, f in enumerate(self.two_body):
        for j in range(n):
            c = (i * n) + j
            f.move2frag(
                to=self.one_body, coeff=float(optimal_coeffs[c]), orb=j, mutate=True
            )


# == Simple Test Cases ==
def greedy_E_optimize(
    ob: OneBodyFragment,
    frags: List[OneBodyFragment],
    iters: int,
    min_eig: Callable,
):
    print(
        f"""starting eigenvalue sum: {
            min_eig(ob.to_tensor())
            + min_eig(frags[0].to_tensor())
            + min_eig(frags[1].to_tensor())
        }"""
    )

    n = frags[0].lambdas.shape[0]
    starting_E = (
        min_eig(ob.to_tensor())
        + min_eig(frags[0].to_tensor())
        + min_eig(frags[1].to_tensor())
    )
    x0 = np.random.random(n * len(frags))

    def cost(x0_0):
        obf_copy = deepcopy(ob)
        frags_copy = deepcopy(frags)
        for i, f in enumerate(frags_copy):
            for j in range(n):
                move_ob_to_ob(from_ob=f, to_ob=obf_copy, coeff=x0_0[(i * n) + j], orb=j)
        new_o_E = min_eig(obf_copy.to_tensor())
        new_f_E = sum([min_eig(f.to_tensor()) for f in frags_copy])
        new_E = new_o_E + new_f_E
        return starting_E - new_E

    coeffs = minimize(
        cost,
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": iters, "disp": False},
    )

    for i, f in enumerate(frags):
        for c in range(n):
            ind = (i * n) + c
            move_ob_to_ob(from_ob=f, to_ob=ob, coeff=coeffs.x[ind], orb=c)


def simple_convex_opt(
    ob: OneBodyFragment,
    frags: List[OneBodyFragment],
    min_eig: Callable,
):
    n = frags[0].lambdas.shape[0]

    num_coeffs = reduce(
        lambda x, y: np.concatenate((x, y), axis=None), [f.lambdas for f in frags]
    )
    coeffs = [cp.Variable() for _ in range(n * len(frags))]
    geq_con = [
        c >= 0 if (num_coeffs[i] > 0) else c >= num_coeffs[i]
        for i, c in enumerate(coeffs)
    ]
    leq_con = [
        c <= num_coeffs[i] if (num_coeffs[i] > 0) else c <= 0
        for i, c in enumerate(coeffs)
    ]
    constraints = geq_con + leq_con
    ob_t = ob.to_tensor()
    f_1 = frags[0].to_tensor()
    f_2 = frags[1].to_tensor()
    print(f"starting eigenvalue sum: {min_eig(ob_t) + min_eig(f_1) + min_eig(f_2)}")
    u_1 = jl_make_u(frags[0].thetas, n)
    u_2 = jl_make_u(frags[1].thetas, n)
    l_1 = cp.diag(cp.vstack(coeffs[0:4]))
    l_2 = cp.diag(cp.vstack(coeffs[4:]))

    fluid_1 = u_1 @ l_1 @ np.linalg.inv(u_1)
    fluid_2 = u_2 @ l_2 @ np.linalg.inv(u_2)

    new_obt = ob_t + fluid_1 + fluid_2
    new_f1 = f_1 - fluid_1
    new_f2 = f_2 - fluid_2

    objective = cp.Maximize(
        cp.lambda_min(new_obt) + cp.lambda_min(new_f1) + cp.lambda_min(new_f2)
    )
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)
    print("optimal value", problem.value)
    optimal_coeffs = [c.value for c in coeffs]
    for i, f in enumerate(frags):
        c = i * n
        for j in range(n):
            move_ob_to_ob(from_ob=f, to_ob=ob, coeff=optimal_coeffs[c + j], orb=j)
