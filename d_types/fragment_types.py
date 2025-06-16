from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openfermion import FermionOperator
import numpy as np

from min_part.f_3_ops import (
    get_obp_from_frag_gfro,
    remove_obt_gfro,
    gfro2fluid,
    tbt2op_gfro,
    move_onebody_coeff_gfro,
    get_obp_from_frag_lr,
    lr2fluid,
    tbtop_lr,
    move_onebody_coeff_lr,
)

Nums = List[int] | List[float] | np.ndarray


@dataclass
class FluidCoeff:
    coeff: float
    thetas: Nums


@dataclass
class FluidParts:
    static_lambdas: Nums
    fluid_lambdas: Nums


@dataclass
class OneBodyFragment:
    fluid_lambdas: List[Tuple[int, FluidCoeff]]
    thetas: Nums
    diag_thetas: Nums
    operators: FermionOperator


@dataclass
class FermionicFragment:
    thetas: Nums
    operators: FermionOperator

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


@dataclass
class GFROFragment(FermionicFragment):
    lambdas: Nums
    fluid_parts: Optional[FluidParts] = None

    def get_ob_lambdas(self):
        """Returns the one-body part from a lambda matrix formed after LR or GFRO decomposition.
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
        return remove_obt_gfro(self)

    def to_fluid(self):
        """Converts GFRO fragment to Fluid

        Args:
            performant: whether or not to perform extra checks. Setting this to True performs operator and equality checks.
            frag: A fragment generated from GFRO or LR procedure.

        Returns:
            A fragment type ready for optimization as a fluid fermionic fragment.
        """
        return gfro2fluid(self)

    def to_op(self):
        """Makes the `FermionOperator` Object from two-body parts for GFRO fragment.

        Args:
            self:

        Returns:
             `FermionOperator` containing only one body parts
        """
        return tbt2op_gfro(self)

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
        raise move_onebody_coeff_gfro(self, to, coeff, orb, mutate)


def remove_obt_lr(self):
    pass


@dataclass
class LRFragment(FermionicFragment):
    coeffs: Nums
    diag_coeffs: Nums
    outer_coeff: float
    fluid_parts: Optional[FluidParts] = None

    def get_ob_lambdas(self):
        return get_obp_from_frag_lr(self)

    def remove_obp(self):
        return remove_obt_lr(self)

    def to_fluid(self):
        return lr2fluid(self)

    def to_op(self):
        return tbtop_lr(self)

    def move2frag(self, to: OneBodyFragment, coeff: float, orb: int, mutate: bool):
        return move_onebody_coeff_lr(self, to, coeff, orb, mutate)


@dataclass
class FragmentedHamiltonian:
    constant: any
    one_body: FermionicFragment
    two_body: List[FermionicFragment]
