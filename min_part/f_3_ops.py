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
    make_lambda_matrix,
    make_unitary,
    extract_thetas,
)
from min_part.julia_ops import solve_quad, eigen_jl
from min_part.lr_decomp import make_unitary_im
from min_part.tensor_utils import tbt2op, obt2op

# == GFRO Helpers


def get_obp_from_frag_gfro(self: GFROFragment) -> np.ndarray:
    n = solve_quad(1, 1, -2 * self.lambdas.size)
    curr_i = 0
    ob = np.ones((n,), dtype=np.float64)
    i = 0
    for j in reversed(range(n)):
        ob[i] = self.lambdas[curr_i]
        curr_i += j + 1
        i += 1
    assert len(ob) == n
    return ob


def remove_obt_gfro(self: GFROFragment) -> Nums:
    n = solve_quad(1, 1, -2 * self.lambdas.size)
    curr_i = 0
    static_frags = np.zeros((self.lambdas.size,))
    skip_indx = []
    for j in reversed(range(n)):
        skip_indx.append(curr_i)
        curr_i += j + 1

    for i, l in enumerate(self.lambdas):
        if i not in skip_indx:
            static_frags[i] = self.lambdas[i]
    return static_frags


def tbt_ob_2ten_gfro(self: GFROFragment) -> np.ndarray:
    if not self.fluid_parts:
        raise UserWarning(
            "Call `to_fluid` method to partition into fluid and static parts!"
        )
    lambdas = np.array(self.fluid_parts.static_lambdas)
    n = solve_quad(1, 1, -2 * lambdas.size)
    l_mat = make_lambda_matrix(lambdas, n)
    unitary = make_unitary(self.thetas, n)
    return contract("lm,lp,lq,mr,ms->pqrs", l_mat, unitary, unitary, unitary, unitary)


def tbt_ob_2op_gfro(self: GFROFragment) -> FermionOperator:
    obt = obp_of_tbp_2t(self.fluid_parts.fluid_lambdas, self.thetas)
    self.operators = tbt2op(obt + tbt_ob_2ten_gfro(self))
    return self.operators


def gfro2fluid(self: GFROFragment, performant: bool = False) -> GFROFragment:
    self.lambdas.setflags(write=False)
    self.thetas.setflags(write=False)
    fluid_frags = self.get_ob_lambdas()
    static_frags = self.remove_obp()
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
def obp_of_tbp_2t(lambdas, thetas) -> np.ndarray:
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
    thetas, diags = extract_thetas(U)
    return OneBodyFragment(
        thetas=thetas,
        diag_thetas=diags,
        lambdas=V,
        fluid_lambdas=[],
        operators=obt2op(obt),
    )


def fluid_ob2ten(self: OneBodyFragment) -> np.ndarray:
    n = self.lambdas.size
    orig_U = make_unitary_im(thetas=self.thetas, diags=self.diag_thetas, n=n)
    h_pq = contract(
        "r,rp,rq->pq",
        self.lambdas,
        orig_U,
        orig_U,
    )
    for orb, fluid_part in self.fluid_lambdas:
        fluid_l = np.zeros((n,))
        fluid_l[orb] = fluid_part.coeff
        unitary = make_unitary(fluid_part.thetas, n)
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
