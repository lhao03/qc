from copy import deepcopy
from typing import List, Tuple

from cvxpy import Variable

import cvxpy as cp

from d_types.config_types import ContractPattern
from min_part.tensor import make_lambda_matrix


def make_fluid_variables(n, self):
    fluid_variables = [cp.Variable() for _ in range(n * len(self.two_body))]
    return fluid_variables


def sum_over_r(lambdas, u_1, u_2):
    """
    Performs the same matrix operation as either r, pr, qr or r, rp, rq, depending on whether u_1/u_2 is a row or a column.
    Args:
        lambdas:
        u_1:
        u_2:

    Returns:

    """
    r_tot = len(lambdas)
    r_sum = 0
    for r in range(r_tot):
        r_sum += lambdas[r] * u_1[r] * u_2[r]
    return r_sum


def make_ob_matrices(contract_pattern, fluid_lambdas: List, self, unitaries):
    fluid_tensors = []
    n = unitaries[0].shape[0]
    for i in range(len(self.two_body)):
        unitary = unitaries[i]
        var_m = []
        for p in range(n):
            row_vars = []
            for q in range(n):
                frag_lambdas = fluid_lambdas[i * n : (i * n) + n]
                if contract_pattern is ContractPattern.LR:
                    row_vars.append(sum_over_r(frag_lambdas, unitary[p], unitary[q]))
                elif contract_pattern is ContractPattern.GFRO:
                    row_vars.append(
                        sum_over_r(frag_lambdas, unitary[:, p], unitary[:, q])
                    )
            var_m.append(cp.hstack(row_vars))

        m = cp.vstack(var_m)
        fluid_tensors.append(m)
    return fluid_tensors


def fluid_ob_op(fluid_tensors, self):
    new_obt = deepcopy(self.one_body.to_tensor())
    for fluid_tensor in fluid_tensors:
        new_obt = new_obt + fluid_tensor
    return new_obt


def summed_fragment_energies(frag_energies, new_obt, self):
    return cp.lambda_sum_smallest(new_obt, self.subspace.expected_e) + cp.sum(
        [cp.min(energy) for energy in frag_energies]
    )


def get_energy_expressions(
    i,
    n,
    num_coeffs: List[float],
    f,
    fluid_variables: List[Variable],
    desired_occs: List[Tuple],
):
    curr_coeffs = num_coeffs[i * n : (i * n) + n]
    curr_variables = fluid_variables[i * n : (i * n) + n]
    lambda_matrix = make_lambda_matrix(f.fluid_parts.static_lambdas, n)
    energies = []
    for occ in desired_occs:
        occ_expression = 0
        for i in occ:
            for j in occ:
                if i == j:
                    occ_expression = occ_expression + (
                        curr_coeffs[i] - curr_variables[i]
                    )
                else:
                    occ_expression = occ_expression + lambda_matrix[i][j]
        energies.append(occ_expression)
    return energies


def tb_energy_expressions(
    desired_occs: List[Tuple],
    fluid_variables: List[Variable],
    n: int,
    num_coeffs: List[float],
    self,
):
    return [
        cp.hstack(
            get_energy_expressions(i, n, num_coeffs, f, fluid_variables, desired_occs)
        )
        for i, f in enumerate(self.two_body)
    ]
