import warnings
from copy import copy
from functools import reduce
from itertools import groupby
from typing import Tuple, Optional, List

import numpy as np
from openfermion import FermionOperator
from opt_einsum import contract

from d_types.fragment_types import (
    FermionicFragment,
    LRFragment,
    GFROFragment,
    Nums,
    FluidCoeff,
    FluidParts,
    OneBodyFragment,
)
from d_types.config_types import ContractPattern, Basis
from min_part.julia_ops import (
    solve_quad,
)
from min_part.tensor import (
    obt2op,
    extract_lambdas,
    make_lambda_matrix,
    make_fr_tensor_from_u,
    get_n_body_tensor_chemist_ordering,
)
from d_types.unitary_type import (
    make_unitary,
    jl_make_u_im,
    jl_make_u,
    ReaDeconUnitary,
    Unitary,
    WholeUnitary,
)

f = np.vectorize(lambda x: x if abs(x) > (np.finfo(float).eps ** 0.5) else 0.0)


def cast_to_real(m):
    return np.real_if_close(m, tol=10000000)


# == GFRO Helpers


def get_obp_from_frag_gfro(self: GFROFragment) -> np.ndarray:
    n = solve_quad(1, 1, -2 * self.lambdas.size)
    curr_i = 0
    ob = np.zeros((n,), dtype=np.float64)
    i = 0
    for j in reversed(range(n)):
        ob[i] = self.lambdas[curr_i]
        curr_i += j + 1
        i += 1
    assert len(ob) == n
    return ob


def remove_obp_gfro(self: GFROFragment) -> Nums:
    n = solve_quad(1, 1, -2 * self.lambdas.size)
    curr_i = 0
    static_frags = np.zeros((self.lambdas.size,), dtype=np.float64)
    skip_indx = []
    for j in reversed(range(n)):
        skip_indx.append(curr_i)
        curr_i += j + 1

    for i, l in enumerate(self.lambdas):
        if i not in skip_indx:
            static_frags[i] = self.lambdas[i]
    return static_frags


def static_2tensor(self: GFROFragment) -> np.ndarray:
    if not self.fluid_parts:
        raise UserWarning(
            "Call `to_fluid` method to partition into fluid and static parts!"
        )
    n = solve_quad(1, 1, -2 * self.fluid_parts.static_lambdas.size)
    unitary = self.unitary.make_unitary_matrix()
    return make_fr_tensor_from_u(self.fluid_parts.static_lambdas, unitary, n=n)


def fluid_gfro_2tensor(self: GFROFragment) -> np.ndarray:
    if isinstance(self.unitary, ReaDeconUnitary):
        obt = fluid_2tensor(self.fluid_parts.fluid_lambdas, self.unitary.thetas)
        return obt + static_2tensor(self)
    else:
        raise UserWarning("Expected a unitary that can be made from real thetas")


def gfro2fluid(self: GFROFragment, performant: bool = False) -> GFROFragment:
    self.lambdas.setflags(write=False)
    fluid_frags = self.get_ob_lambdas()
    static_frags = self.remove_obp()
    static_frags = f(static_frags)
    fluid_frags = f(fluid_frags)
    self.fluid_parts = FluidParts(
        static_lambdas=static_frags, fluid_lambdas=fluid_frags
    )
    self.fluid_parts.static_lambdas.setflags(write=False)
    if not performant:
        assert self.operators == self.to_op()
    return self


def move_onebody_coeff(
    self: FermionicFragment,
    to: OneBodyFragment,
    coeff: float,
    orb: int,
    mutate: bool = True,
) -> Optional[Tuple[FermionicFragment, OneBodyFragment]]:
    if not mutate:
        raise NotImplementedError
    if not self.fluid_parts:
        raise UserWarning(
            "Fragment is not fluid yet! Call `to_fluid` method to partition into fluid and static parts!"
        )
    assert orb <= self.fluid_parts.fluid_lambdas.size
    assert coeff.imag == 0
    self.fluid_parts.fluid_lambdas[orb] -= coeff
    to.fluid_lambdas.append(
        (
            orb,
            FluidCoeff(
                coeff=coeff,
                unitary=self.unitary,
                contract_pattern=ContractPattern.GFRO
                if isinstance(self, GFROFragment)
                else ContractPattern.LR,
            ),
        )
    )
    return self, to


def fluid_2tensor(lambdas, thetas) -> np.ndarray:
    """Makes tensor using one-body parts that used to below in a two-electron fragment.

    Returns:
         `FermionOperator` containing only one body parts
    """
    diags = np.array(lambdas)
    n = diags.size
    l_mat = np.diagflat(diags)
    unitary = make_unitary(thetas, n)
    return contract("lm,lp,lq,mr,ms->pqrs", l_mat, unitary, unitary, unitary, unitary)


# == LR Helpers
def get_obp_from_frag_lr(self: LRFragment):
    lambdas = self.fluid_parts.static_lambdas
    n = self.coeffs.size
    curr_i = 0
    ob = np.zeros((n,), dtype=np.float64)
    i = 0
    for j in reversed(range(n)):
        ob[i] = lambdas[curr_i]
        curr_i += j + 1
        i += 1
    assert len(ob) == n
    return ob


def remove_obp_lr(self: LRFragment):
    lambdas = self.fluid_parts.static_lambdas
    n = self.coeffs.size
    curr_i = 0
    static_frags = np.zeros((lambdas.size,), dtype=np.float64)
    skip_indx = []
    for j in reversed(range(n)):
        skip_indx.append(curr_i)
        curr_i += j + 1

    for i, l in enumerate(lambdas):
        if i not in skip_indx:
            static_frags[i] = lambdas[i]
    return static_frags


def fluid_lr_2tensor(self: LRFragment) -> np.ndarray:
    diags = np.array(self.fluid_parts.fluid_lambdas)
    n = diags.size
    l_mat = np.diagflat(diags)
    unitary = self.unitary.make_unitary_matrix()
    constract_pattern = "lm,pl,ql,rm,sm->pqrs"  # TODO: check??
    tbt_obp = contract(constract_pattern, l_mat, unitary, unitary, unitary, unitary)
    static_l_mat = make_lambda_matrix(self.fluid_parts.static_lambdas, n)
    tbt = contract(constract_pattern, static_l_mat, unitary, unitary, unitary, unitary)
    return tbt_obp + tbt


def lr2fluid(self: LRFragment, performant: bool = False) -> LRFragment:
    tensor_to_check = (
        None
        if performant
        else get_n_body_tensor_chemist_ordering(self.operators, n=2, m=8)
    )
    self.coeffs.setflags(write=False)
    n = self.coeffs.size
    c = np.reshape(self.coeffs, (n, 1))
    c_matrix = self.outer_coeff * c @ c.T
    lambdas = extract_lambdas(c_matrix, n)
    self.fluid_parts = FluidParts(static_lambdas=lambdas, fluid_lambdas=None)
    fluid_frags = self.get_ob_lambdas()
    static_frags = self.remove_obp()
    tol = np.finfo(float).eps ** 0.5
    f = np.vectorize(lambda x: x if abs(x) > tol else 0.0)
    self.fluid_parts.fluid_lambdas = f(fluid_frags)
    self.fluid_parts.static_lambdas = f(static_frags)
    self.fluid_parts.static_lambdas.setflags(write=False)
    assert self.fluid_parts.fluid_lambdas.size == n
    if not performant:
        np.testing.assert_array_almost_equal(tensor_to_check, self.to_tensor())
    return self


# === One Body Helpers
def move_ob_to_ob(
    from_ob: OneBodyFragment, to_ob: OneBodyFragment, coeff: float, orb: int
):
    """
    Move a one-body part of a one-electron operator to another one-electron operator.
    For testing the optimization purposes.

    Args:
        from_ob:
        to_ob:
        coeff:
        orb:

    Returns:

    """
    from_ob.lambdas[orb] -= coeff
    to_ob.fluid_lambdas.append(
        (
            orb,
            FluidCoeff(
                coeff=coeff,
                contract_pattern=ContractPattern.LR,
                unitary=from_ob.unitary,
            ),
        )
    )


def obt2fluid(obt: np.ndarray, basis: Basis) -> OneBodyFragment:
    """
    Converts a one-body tensor to a `OneBodyFragment` type via diagonalization of the tensor.
    Args:
        obt:

    Returns:
        `FluidFermionicFragment` object of the one-body tensor
    """
    if isinstance(obt, OneBodyFragment):
        return obt
    V, U = np.linalg.eigh(obt)
    assert V.size == obt.shape[0]
    assert U.shape == obt.shape
    swapped = False
    try:
        if np.isclose(np.linalg.det(U), -1):
            U[:, [0, 1]] = U[:, [1, 0]]
            prev_0 = V[0]
            prev_1 = V[1]
            V[0] = prev_1
            V[1] = prev_0
            swapped = True
        unitary = Unitary.deconstruct_unitary(U, basis=basis)
        made_u = unitary.make_unitary_matrix()
        np.testing.assert_array_almost_equal(made_u, U)
        return OneBodyFragment(
            unitary=unitary,
            lambdas=V,
            fluid_lambdas=[],
            operators=obt2op(obt),
        )
    except Exception as e:
        warnings.warn(
            f"Tried to decompose unitary when preparing one body matrix for fluid, didn't work so storing entire unitary.: {e}"
        )
        if swapped:
            U[:, [0, 1]] = U[:, [1, 0]]
            prev_0 = V[0]
            prev_1 = V[1]
            V[0] = prev_1
            V[1] = prev_0
        return OneBodyFragment(
            unitary=WholeUnitary(mat=U, dim=obt.shape[0], basis=basis),
            lambdas=V,
            fluid_lambdas=[],
            operators=obt2op(obt) if basis == Basis.SPIN else None,
        )


def fluid_ob2ten(self: OneBodyFragment) -> np.ndarray:
    n = self.lambdas.size
    unitary = self.unitary.make_unitary_matrix()
    h_pq = contract(
        "r,pr,qr->pq",
        self.lambdas,
        unitary,
        unitary,
    )
    for orb, fluid_part in self.fluid_lambdas:
        fluid_h = make_obp_tensor(fluid_part, n, orb)
        h_pq += fluid_h
    return h_pq


def make_unitary_py(n, self):
    return (
        self.unitary
        if isinstance(self.unitary, np.ndarray)
        else (make_unitary_jl(n, self))
    )


def make_unitary_jl(n, self):
    if not hasattr(self, "diag_thetas"):
        return jl_make_u(self.thetas, n)
    else:
        return (
            jl_make_u_im(self.thetas, self.diag_thetas, n)
            if isinstance(self.diag_thetas, np.ndarray)
            else jl_make_u(self.thetas, n)
        )


def make_obp_tensor(fluid_part: FluidCoeff, n: int, orb: int):
    fluid_l = np.zeros((n,), dtype=np.float64)
    fluid_l[orb] = fluid_part.coeff
    unitary = fluid_part.unitary.make_unitary_matrix()
    fluid_h = contract(
        fluid_part.contract_pattern.value,
        fluid_l,
        unitary,
        unitary,
    )
    fluid_h = cast_to_real(fluid_h)
    return fluid_h


def fluid_ob2op(self: OneBodyFragment) -> FermionOperator:
    """Rediagonalization of the one body fragment, according to https://quantum-journal.org/papers/q-2023-01-03-889/pdf/.

    whwere h_{pq}' = h_{pq} + sum of every U c U^T

    Args:
        undiagonalized_onebody

    Returns:
        the modified fluid one body fragment
    """
    self.operators = obt2op(fluid_ob2ten(self))
    return self.operators


def get_diag_idx(orb, n) -> int:
    m = (n * (n + 1)) // 2
    return m - reduce(lambda a, b: a + b, range((n - orb) + 1))


def make_lambdas(fluid_coeffs: List[Tuple[int, FluidCoeff]], n):
    m = (n * (n + 1)) // 2
    lambdas = np.zeros((m,))
    grouped_lambdas = groupby(fluid_coeffs, lambda x: x[0])
    for orb, group in grouped_lambdas:
        c = 0
        for g in group:
            c += g[1].coeff
        idx = get_diag_idx(orb, n)
        lambdas[idx] = c
    return lambdas


def lambdas_from_fluid_parts(fluid_parts: FluidParts):
    m = fluid_parts.static_lambdas.size
    n = solve_quad(1, 1, -2 * m)
    lambdas = copy(fluid_parts.static_lambdas)
    for i, c in enumerate(fluid_parts.fluid_lambdas):
        lambdas[get_diag_idx(i, n)] = c
    return lambdas
