import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
from numpy import ndarray, dtype
from openfermion import (
    FermionOperator,
    is_hermitian,
)

from d_types.config_types import (
    Nums,
    ContractPattern,
    Basis,
)
from d_types.unitary_type import Unitary

from min_part.julia_ops import jl_print
from min_part.operators import (
    generate_occupied_spin_orb_permutations,
)

from min_part.tensor import (
    tbt2op,
    get_n_body_tensor_chemist_ordering,
    make_lambda_matrix,
    extract_lambdas,
    make_fr_tensor_from_u,
)


@dataclass
class FluidCoeff:
    coeff: float
    contract_pattern: ContractPattern
    unitary: Unitary


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
    operators: Optional[FermionOperator]
    unitary: Unitary

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
            return (
                self.operators == other.operators
                and self.fluid_lambdas == other.fluid_lambdas
                and np.allclose(self.lambdas, other.lambdas)
                and u_eq
            )

    def spin2spac(self):
        raise NotImplementedError

    def spac2spin(self):
        raise NotImplementedError

    def to_op(self):
        ob_op = fluid_ob2op(self)
        if is_hermitian(ob_op):
            return ob_op
        else:
            mat = get_n_body_tensor_chemist_ordering(ob_op, n=1, m=4)
            jl_print(mat)
            is_her_mat = np.allclose(mat, mat.T)
            warnings.warn(
                f"Operator may not be Hermitian according to OpenFermion, np says: {is_her_mat}"
            )
            return ob_op

    def to_tensor(self):
        return fluid_ob2ten(self)

    def get_expectation_value(self, elecs):
        vals, vecs = np.linalg.eigh(self.to_tensor())
        occupation_combinations = generate_occupied_spin_orb_permutations(
            self.lambdas.shape[0], elecs
        )
        energies = []
        for occ in occupation_combinations:
            energy = 0
            for i in occ:
                energy += vals[i]
            energies.append((occ, energy))
        return energies


@dataclass(kw_only=True)
class FermionicFragment:
    unitary: Unitary
    basis: Basis
    operators: Optional[FermionOperator] = None
    fluid_parts: Optional[FluidParts] = None

    @abstractmethod
    def spin2spac(self):
        raise NotImplementedError

    @abstractmethod
    def spac2spin(self):
        raise NotImplementedError

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

    def bulkmove2frag(self, to: OneBodyFragment, coeffs: Nums):
        for i, c in enumerate(coeffs):
            self.move2frag(to, c, i, True)

    @abstractmethod
    def to_tensor(self):
        raise NotImplementedError

    @abstractmethod
    def get_expectation_value(self, num_spin_orbs: int, expected_e: int):
        raise NotImplementedError


@dataclass(kw_only=True)
class GFROFragment(FermionicFragment):
    lambdas: Nums

    def __eq__(self, other):
        if isinstance(other, GFROFragment):
            return (
                np.allclose(self.lambdas, other.lambdas)
                and self.operators == other.operators
                and self.fluid_parts == other.fluid_parts
                and self.unitary == other.unitary
            )
        else:
            return False

    def spac2spin(self):
        if self.basis == Basis.SPATIAL:
            num_spatial = self.unitary.dim
            spatial_l_m = make_lambda_matrix(self.lambdas, num_spatial)
            spin_lambdas = np.zeros((num_spatial * 2, num_spatial * 2))
            for i in range(1, num_spatial + 1):
                for j in range(1, num_spatial + 1):
                    i2 = 2 * i
                    j2 = 2 * j
                    i2min1 = i2 - 1
                    j2min1 = j2 - 1
                    tilde_l_ij = spatial_l_m[i - 1, j - 1]
                    spin_lambdas[i2 - 1, j2 - 1] = tilde_l_ij
                    spin_lambdas[i2min1 - 1, j2 - 1] = tilde_l_ij
                    spin_lambdas[i2 - 1, j2min1 - 1] = tilde_l_ij
                    spin_lambdas[i2min1 - 1, j2min1 - 1] = tilde_l_ij
            self.lambdas = extract_lambdas(spin_lambdas, num_spatial * 2)
            self.unitary = self.unitary.spac2spin()
            self.basis = Basis.SPIN
            self.operators = self.to_op()
        return self

    def spin2spac(self):
        warnings.warn("Will fail if symmetries aren't seen.")
        if self.basis == Basis.SPIN:
            num_spin = self.unitary.dim
            spin_lm = make_lambda_matrix(self.lambdas, num_spin)
            num_spat = num_spin // 2
            spat_lambdas = np.zeros((num_spat, num_spat))
            for i in range(1, num_spat + 1):
                for j in range(1, num_spat + 1):
                    i2 = 2 * i
                    j2 = 2 * j
                    i2min1 = i2 - 1
                    j2min1 = j2 - 1
                    l_1 = spin_lm[i2 - 1, j2 - 1]
                    l_2 = spin_lm[i2min1 - 1, j2 - 1]
                    l_3 = spin_lm[i2 - 1, j2min1 - 1]
                    l_4 = spin_lm[i2min1 - 1, j2min1 - 1]
                    if np.isclose(l_1, l_2) and np.isclose(l_3, l_4):
                        spat_lambdas[i - 1, j - 1] = l_1
                    else:
                        raise UserWarning(
                            f"Expected symmetries, didn't see, got: {l_1},"
                            f" {l_2}, "
                            f"{l_3},"
                            f" {l_4}"
                        )
            self.unitary = self.unitary.spin2spac()
            self.lambdas = extract_lambdas(spat_lambdas, num_spat)
            self.basis = Basis.SPATIAL
            self.operators = None
        return self

    def get_expectation_value(self, num_spin_orbs: int, expected_e: int):
        if self.fluid_parts is None:
            return get_expectation_vals_gfro(self, num_spin_orbs, expected_e)
        else:
            self.lambdas = lambdas_from_fluid_parts(self.fluid_parts)
            return get_expectation_vals_gfro(self, num_spin_orbs, expected_e)

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
        if not self.fluid_parts:
            return make_fr_tensor_from_u(
                self.lambdas, self.unitary.make_unitary_matrix(), self.unitary.dim
            )
        return fluid_gfro_2tensor(self)

    def to_op(self):
        """Makes the `FermionOperator` Object of the fluid GFROFragment, summing together one and two body parts.

        Args:
            self:

        Returns:
             `FermionOperator` containing "one" (the one body part from the two body fragment) and two body parts
        """
        if self.operators is None:
            self.operators = tbt2op(self.to_tensor())
        if self.fluid_parts:
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
        if not (isinstance(coeff, float) or isinstance(coeff, int)):
            raise UserWarning(f"Got type {type(coeff)}")
        return move_onebody_coeff(self, to, coeff, orb, mutate)


def gfro_fragment_occ(
    fragment: GFROFragment, num_spin_orbs: int, occ: Optional[int]
) -> Tuple[list[Tuple[int]], ndarray[Any, dtype[Any]]]:
    """Given a fragment generated by GFRO, determine the energy of the fragment for all possible electron
    occupation configurations. Assumes `openfermion` spin orbital numbering, where even numbers are spin up, and
    odd numbers are spin down.

    Args:
        fragment: a fragment in the GFRO fragment form
        num_spin_orbs: number of all orbitals (the alpha and beta orbitals count as 2)

    Returns:
        energies of the fragment given a certain occupation of spin orbitals
    """
    lambda_matrix = make_lambda_matrix(fragment.lambdas, num_spin_orbs)
    occupation_combinations = generate_occupied_spin_orb_permutations(
        num_spin_orbs, occ
    )
    occ_energies = []
    for occ_comb in occupation_combinations:
        occ_energy = 0
        for l in occ_comb:
            for m in occ_comb:
                occ_energy += lambda_matrix[l][m]
        occ_energies.append(occ_energy)
    return occupation_combinations, np.array(occ_energies)


def get_expectation_vals_gfro(self: GFROFragment, num_spin_orbs: int, expected_e: int):
    return gfro_fragment_occ(
        fragment=self,
        num_spin_orbs=num_spin_orbs,
        occ=expected_e,
    )


@dataclass(kw_only=True)
class LRFragment(FermionicFragment):
    coeffs: Nums
    unitary: Unitary
    outer_coeff: float

    def __eq__(self, other):
        if isinstance(other, LRFragment):
            return (
                np.allclose(self.coeffs, other.coeffs)
                and self.unitary == other.unitary
                and self.outer_coeff == other.outer_coeff
                and self.operators == other.operators
                and self.fluid_parts == other.fluid_parts
            )
        else:
            return False

    def spac2spin(self):
        raise NotImplementedError

    def spin2spac(self):
        raise NotImplementedError

    def get_expectation_value(self, num_spin_orbs: int, expected_e: int):
        return get_expectation_vals_lr_frags(self, num_spin_orbs, expected_e)

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
        return lr2fluid(self, performant=performant)

    def to_op(self):
        if self.fluid_parts is None:
            return self.operators
        self.operators = tbt2op(fluid_lr_2tensor(self))
        return self.operators

    def move2frag(self, to: OneBodyFragment, coeff: float, orb: int, mutate: bool):
        return move_onebody_coeff(self, to, coeff, orb, mutate)


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
    lambdas_from_fluid_parts,
)

from min_part.lr_decomp import get_expectation_vals_lr_frags  # noqa: E402
