from copy import deepcopy
from functools import reduce
from typing import List, Callable, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult
import cvxpy as cp

from d_types.config_types import ContractPattern
from d_types.cvx_exp import (
    make_fluid_variables,
    make_ob_matrices,
    fluid_ob_op,
    tb_energy_expressions,
    summed_fragment_energies,
)
from d_types.fragment_types import (
    OneBodyFragment,
    FermionicFragment,
    GFROFragment,
)
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_opers import move_ob_to_ob
from d_types.unitary_type import jl_make_u


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


def convex_optimization(self: FragmentedHamiltonian, desired_occs: List[Tuple]):
    n = self.one_body.lambdas.shape[0]
    num_coeffs = []
    for f in self.two_body:
        f.to_fluid()
        num_coeffs = np.append(num_coeffs, f.fluid_parts.fluid_lambdas)

    fluid_variables = make_fluid_variables(n, self)
    constraints = [
        c >= 0 if (num_coeffs[i] > 0) else c >= 100 * num_coeffs[i]
        for i, c in enumerate(fluid_variables)
    ] + [
        c <= num_coeffs[i] if (num_coeffs[i] > 0) else c <= 0
        for i, c in enumerate(fluid_variables)
    ]
    unitaries = [f.unitary.make_unitary_matrix() for f in self.two_body]
    tb_frag_energies = tb_energy_expressions(
        desired_occs, fluid_variables, n, num_coeffs, self
    )
    constract_pattern = (
        ContractPattern.GFRO
        if isinstance(self.two_body[0], GFROFragment)
        else ContractPattern.LR
    )
    ob_fluid_parts = make_ob_matrices(
        contract_pattern=constract_pattern,
        fluid_lambdas=fluid_variables,
        self=self,
        unitaries=unitaries,
    )
    new_obt = fluid_ob_op(ob_fluid_parts, self)
    objective = cp.Maximize(
        summed_fragment_energies(
            frag_energies=tb_frag_energies, new_obt=new_obt, self=self
        )
    )
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_coeffs: List[float] = [float(c.value) for c in fluid_variables]
    for i, f in enumerate(self.two_body):
        f.bulkmove2frag(to=self.one_body, coeffs=optimal_coeffs[i * n : (i * n) + n])


# == Simple Test Cases ==


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
