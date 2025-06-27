from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
from openfermion import FermionOperator

from d_types.config_types import Nums

from min_part.tensor import tbt2op


class PartitionStrategy(Enum):
    GFRO = "GFRO"
    LR = "LR"


class ContractPattern(Enum):
    GFRO = "r,rp,rq->pq"
    LR = "r,pr,qr->pq"


@dataclass
class FluidCoeff:
    coeff: float
    thetas: Nums
    contract_pattern: ContractPattern
    diag_thetas: Optional[Nums] = None


@dataclass
class FluidParts:
    static_lambdas: Nums
    fluid_lambdas: Nums

    def __eq__(self, other):
        if isinstance(other, FluidParts):
            return np.allclose(self.fluid_lambdas, other.fluid_lambdas) and np.allclose(
                self.static_lambdas, other.static_lambdas
            )
        else:
            return False


@dataclass
class OneBodyFragment:
    fluid_lambdas: List[Tuple[int, FluidCoeff]]
    lambdas: Nums
    thetas: Nums
    operators: FermionOperator
    diag_thetas: Optional[Nums] = None
    unitary: Optional[Nums] = None

    def __eq__(self, other):
        if isinstance(other, OneBodyFragment):
            u_eq = (
                np.allclose(self.unitary, other.unitary)
                if (
                    isinstance(self.unitary, np.ndarray)
                    and isinstance(other.unitary, np.ndarray)
                )
                else self.unitary is None and other.unitary is None
            )
            d_t_eq = (
                np.allclose(self.diag_thetas, other.diag_thetas)
                if (
                    isinstance(self.diag_thetas, np.ndarray)
                    and isinstance(other.diag_thetas, np.ndarray)
                )
                else self.diag_thetas is None and other.diag_thetas is None
            )
            return (
                self.operators == other.operators
                and self.fluid_lambdas == other.fluid_lambdas
                and np.allclose(self.lambdas, other.lambdas)
                and np.allclose(self.thetas, other.thetas)
                and u_eq
                and d_t_eq
            )

    def to_op(self):
        return fluid_ob2op(self)

    def to_tensor(self):
        return fluid_ob2ten(self)


@dataclass(kw_only=True)
class FermionicFragment:
    thetas: Nums
    operators: FermionOperator
    fluid_parts: Optional[FluidParts] = None

    @abstractmethod
    def get_ob_lambdas(self):
        raise NotImplementedError

    @abstractmethod
    def remove_obp(self):
        raise NotImplementedError

    @abstractmethod
    def to_fluid(self):
        raise NotImplementedError

    @abstractmethod
    def to_op(self):
        raise NotImplementedError

    @abstractmethod
    def move2frag(self, to: OneBodyFragment, coeff: float, orb: int, mutate: bool):
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self):
        raise NotImplementedError


@dataclass(kw_only=True)
class GFROFragment(FermionicFragment):
    lambdas: Nums

    def __eq__(self, other):
        if isinstance(other, GFROFragment):
            return (
                np.allclose(self.lambdas, other.lambdas)
                and np.allclose(self.thetas, other.thetas)
                and self.operators == other.operators
                and self.fluid_parts == other.fluid_parts
            )
        else:
            return False

    def get_ob_lambdas(self):
        """Returns the one-body part from a lambda matrix formed after GFRO decomposition.
        Args:
            self: a GFRO fragment

        Returns:
            the one body coefficients in ascending order, 1, 2, 3...n
        """
        return get_obp_from_frag_gfro(self)

    def remove_obp(self):
        """Given a GFRO fragment, remove all one body parts from the lambda array, which is used to form the lambda matrix.
        This procedure assumes a GFRO fragment works

        """
        return remove_obp_gfro(self)

    def to_fluid(self, performant: bool = True):
        """Converts GFRO fragment to Fluid

        Args:
            performant: whether or not to perform extra checks. Setting this to True performs operator and equality checks.
            self: A fragment generated from GFRO procedure.

        Returns:
            A fragment type ready for optimization as a fluid fermionic fragment.
        """
        return gfro2fluid(self, performant=performant)

    def to_tensor(self):
        return fluid_gfro_2tensor(self)

    def to_op(self):
        """Makes the `FermionOperator` Object of the fluid GFROFragment, summing together one and two body parts.

        Args:
            self:

        Returns:
             `FermionOperator` containing "one" (the one body part from the two body fragment) and two body parts
        """
        self.operators = tbt2op(fluid_gfro_2tensor(self))
        return self.operators

    def move2frag(self, to: OneBodyFragment, coeff: float, orb: int, mutate: bool):
        """Moves any real float amount of the one-body coeffcient from a two-electron fragment to a one-body fragment.
        This is done by subtracting the value from the two-electron fragment to the one-electron fragment.

        Args:
            self: the GFRO fragment to alter
            to: the one body fragment to move to
            coeff: the value to remove, must be real (?)
            orb: which orbital to move coeff from, must be less than number of orbitals
            mutate: if set to True, mutates both `self` and `to` in place, else copies and returns new versions

        Returns:
            the mutated/new GFROFragment and OneBodyFragment(no
        """
        assert isinstance(coeff, float) or isinstance(coeff, int)
        return move_onebody_coeff(self, to, coeff, orb, mutate)


@dataclass(kw_only=True)
class LRFragment(FermionicFragment):
    coeffs: Nums
    diag_thetas: Nums
    outer_coeff: float

    def __eq__(self, other):
        if isinstance(other, LRFragment):
            return (
                np.allclose(self.coeffs, other.coeffs)
                and np.allclose(self.thetas, other.thetas)
                and np.allclose(self.diag_thetas, other.diag_thetas)
                and self.outer_coeff == other.outer_coeff
                and self.operators == other.operators
                and self.fluid_parts == other.fluid_parts
            )
        else:
            return False

    def to_tensor(self):
        return fluid_lr_2tensor(self)

    def get_ob_lambdas(self):
        """Returns the one-body part from a lambda matrix formed after LR decomposition.
        Args:
            self: an LR fragment

        Returns:
            the one body coefficients in ascending spin orbital order, 1, 2, 3...n
        """
        return get_obp_from_frag_lr(self)

    def remove_obp(self):
        return remove_obp_lr(self)

    def to_fluid(self, performant: bool = True):
        """Converts an LRFragment into fluid form, by separating out the one-body part from the two-body part,
        if possible.

        Args:
            performant: whether or not to perform extra checks. Setting this to True performs operator and equality checks.
            self: A fragment generated from LR procedure.

        Returns:
            A fragment type ready for optimization as a fluid fermionic fragment.
        """
        return lr2fluid(self)

    def to_op(self):
        self.operators = tbt2op(fluid_lr_2tensor(self))
        return self.operators

    def move2frag(self, to: OneBodyFragment, coeff: float, orb: int, mutate: bool):
        return move_onebody_coeff(self, to, coeff, orb, mutate)


@dataclass
class Subspace:
    n: partial[[np.ndarray], float]
    expected_e: int
    s2: partial[[np.ndarray], float]
    expected_s2: int
    sz: partial[[np.ndarray], float]
    expected_sz: int

    def __eq__(self, other):
        if isinstance(other, Subspace):
            n_eq = self.n.func == other.n.func
            s2_eq = self.s2.func == other.s2.func
            sz_eq = self.sz.func == other.sz.func
            return (
                n_eq
                and s2_eq
                and sz_eq
                and self.expected_e == other.expected_e
                and self.expected_s2 == other.expected_s2
                and self.expected_sz == other.expected_sz
            )
        else:
            return False


# == For importing functions and avoiding circular import error ==
from min_part.f3_opers import (  # noqa: E402
    get_obp_from_frag_gfro,
    remove_obp_gfro,
    gfro2fluid,
    fluid_gfro_2tensor,
    move_onebody_coeff,
    get_obp_from_frag_lr,
    lr2fluid,
    fluid_ob2op,
    fluid_ob2ten,
    fluid_lr_2tensor,
    remove_obp_lr,
)
