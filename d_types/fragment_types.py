import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from openfermion import (
    FermionOperator,
    is_hermitian,
)

from d_types.config_types import (
    Nums,
    ContractPattern,
)

from min_part.julia_ops import jl_print
from min_part.operators import (
    generate_occupied_spin_orb_permutations,
)

from min_part.tensor import tbt2op, get_n_body_tensor_chemist_ordering


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

    def get_expectation_value_cvxpy(self, ham):
        n = self.lambdas.shape[0]
        unitaries = [make_unitary_jl(n=n, self=f) for f in ham.two_body]
        ob_fluid_matrices = make_ob_matrices(
            contract_pattern=self.fluid_lambdas[0][1].contract_pattern,
            fluid_lambdas=ham.fluid_variables,
            self=ham,
            unitaries=unitaries,
        )
        ob_fluid_tensor = fluid_ob_op(ob_fluid_matrices, ham)
        return np.linalg.eigh(ob_fluid_tensor)


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

    def bulkmove2frag(self, to: OneBodyFragment, coeffs: Nums):
        for i, c in enumerate(coeffs):
            self.move2frag(to, c, i, True)

    @abstractmethod
    def to_tensor(self):
        raise NotImplementedError

    @abstractmethod
    def get_expectation_value(self, num_spin_orbs: int, expected_e: int):
        raise NotImplementedError

    @abstractmethod
    def get_expectation_value_cvxpy(
        self,
        num_spin_orbs: int,
        expected_e: int,
        ham,
        desired_occs,
    ):
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

    def get_expectation_value_cvxpy(
        self,
        num_spin_orbs: int,
        expected_e: int,
        ham,
        desired_occs,
    ):
        n = ham.one_body.shape[0]
        num_coeffs = [self.lambdas[get_diag_idx(i, n)] for i in range(n)]
        energy_expressions = get_energy_expressions(
            i=0,
            n=n,
            num_coeffs=num_coeffs,
            f=ham.two_body[0],
            fluid_variables=ham._return_proper_fluid_vars(self),
            desired_occs=desired_occs,
        )
        return [exp.value for exp in energy_expressions]

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
        return fluid_gfro_2tensor(self)

    def to_op(self):
        """Makes the `FermionOperator` Object of the fluid GFROFragment, summing together one and two body parts.

        Args:
            self:

        Returns:
             `FermionOperator` containing "one" (the one body part from the two body fragment) and two body parts
        """
        if self.fluid_parts is None:
            return self.operators
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
        return lr2fluid(self)

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
    make_unitary_jl,
    get_diag_idx,
)

from min_part.gfro_decomp import get_expectation_vals_gfro  # noqa: E402
from min_part.lr_decomp import get_expectation_vals_lr_frags  # noqa: E402

from d_types.cvx_exp import (  # noqa: E402
    fluid_ob_op,
    make_ob_matrices,
    get_energy_expressions,
)
