from typing import Tuple, Optional

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
from min_part.gfro_decomp import (
    make_fr_tensor_from_u,
)
from min_part.julia_ops import solve_quad, jl_make_u_im, jl_make_u, jl_extract_thetas
from min_part.tensor import obt2op, make_unitary


# == GFRO Helpers


def get_obp_from_frag_gfro(self: GFROFragment) -> np.ndarray:
    n = solve_quad(1, 1, -2 * self.lambdas.size)
    curr_i = 0
    ob = np.zeros((n,), dtype=np.float64)
    i = 0
    for j in reversed(range(n)):
        l = self.lambdas[curr_i]
        ob[i] = l
        curr_i += j + 1
        i += 1
    assert len(ob) == n
    return ob


def remove_obt_gfro(self: GFROFragment) -> Nums:
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
    unitary = make_unitary(self.thetas, n)
    return make_fr_tensor_from_u(self.fluid_parts.static_lambdas, unitary, n=n)


def fluid_gfro_2tensor(self: GFROFragment) -> np.ndarray:
    obt = fluid_2tensor(self.fluid_parts.fluid_lambdas, self.thetas)
    return obt + static_2tensor(self)


def gfro2fluid(self: GFROFragment, performant: bool = False) -> GFROFragment:
    self.lambdas.setflags(write=False)
    self.thetas.setflags(write=False)
    fluid_frags = self.get_ob_lambdas()
    static_frags = self.remove_obp()
    tol = np.finfo(float).eps ** 0.5
    f = np.vectorize(lambda x: x if abs(x) > tol else 0.0)
    static_frags = f(static_frags)
    fluid_frags = f(fluid_frags)
    self.fluid_parts = FluidParts(
        static_lambdas=static_frags, fluid_lambdas=fluid_frags
    )
    self.fluid_parts.static_lambdas.setflags(write=False)
    if not performant:
        assert self.operators == self.to_op()
    return self


def move_onebody_coeff_gfro(
    self: GFROFragment,
    to: OneBodyFragment,
    coeff: float,
    orb: int,
    mutate: bool = True,
) -> Optional[Tuple[GFROFragment, OneBodyFragment]]:
    if not mutate:
        raise NotImplementedError
    if not self.fluid_parts:
        raise UserWarning(
            "Call `to_fluid` method to partition into fluid and static parts!"
        )
    assert orb <= self.fluid_parts.fluid_lambdas.size
    assert coeff.imag == 0
    self.fluid_parts.fluid_lambdas[orb] -= coeff
    to.fluid_lambdas.append((orb, FluidCoeff(coeff=coeff, thetas=self.thetas)))
    return self, to


# == LR Helpers


def get_obp_from_frag_lr(self: LRFragment):
    raise NotImplementedError


def remove_obp_lr(self: LRFragment):
    raise NotImplementedError


def tbtop_lr(self: LRFragment) -> LRFragment:
    self.fluid_parts = FluidParts(static_lambdas=[], fluid_lambdas=[])
    raise NotImplementedError
    return self


def lr2fluid(self: LRFragment, performant: bool = False) -> LRFragment:
    """Converts any FermionicFragment into a FluidFermionicFragment, by separating out the one-body part from the two-body part,
    if possible.

    Args:
        performant: whether or not to perform extra checks. Setting this to True performs operator and equality checks.
        frag: A fragment generated from GFRO or LR procedure.

    Returns:
        A fragment type ready for optimization as a fluid fermionic fragment.
    """
    pass


def move_onebody_coeff_lr(
    self: LRFragment,
    to: OneBodyFragment,
    coeff: float,
    orb: int,
    mutate: bool = True,
) -> Optional[Tuple[LRFragment, FermionicFragment]]:
    """Moves any real float amount of the one-body coeffcient from a two-electron fragment to a one-body fragment"""
    pass


# == Two body helpers
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


# === One Body Helpers
def obt2fluid(obt: np.ndarray) -> OneBodyFragment:
    """
    Converts a one-body tensor to a `OneBodyFragment` type via diagonalization of the tensor.
    Args:
        obt:

    Returns:
        `FluidFermionicFragment` object of the one-body tensor
    """
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
        thetas, diags = jl_extract_thetas(U)
        return OneBodyFragment(
            thetas=thetas,
            diag_thetas=None if np.allclose(diags, np.zeros((obt.shape[0]))) else diags,
            lambdas=V,
            fluid_lambdas=[],
            operators=obt2op(obt),
        )
    except RuntimeError as e:
        print(e)
        if swapped:
            U[:, [0, 1]] = U[:, [1, 0]]
            prev_0 = V[0]
            prev_1 = V[1]
            V[0] = prev_1
            V[1] = prev_0
        return OneBodyFragment(
            unitary=U, lambdas=V, fluid_lambdas=[], operators=obt2op(obt), thetas=None
        )


def fluid_ob2ten(self: OneBodyFragment) -> np.ndarray:
    n = self.lambdas.size
    unitary = (
        self.unitary
        if isinstance(self.unitary, np.ndarray)
        else (
            jl_make_u_im(self.thetas, self.diag_thetas, n)
            if isinstance(self.diag_thetas, np.ndarray)
            else jl_make_u(self.thetas, n)
        )
    )
    h_pq = contract(
        "r,pr,qr->pq",
        self.lambdas,
        unitary,
        unitary,
    )
    for orb, fluid_part in self.fluid_lambdas:
        fluid_l = np.zeros((n,))
        fluid_l[orb] = fluid_part.coeff
        unitary = (
            jl_make_u_im(fluid_part.thetas, fluid_part.diag_thetas, n)
            if isinstance(fluid_part.diag_thetas, np.ndarray)
            else jl_make_u(fluid_part.thetas, n)
        )
        fluid_h = contract(
            "r,rp,rq->pq",
            fluid_l,
            unitary,
            unitary,
        )
        h_pq += fluid_h
    return h_pq


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
