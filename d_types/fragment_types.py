from ast import Num
from dataclasses import dataclass
from typing import List

from openfermion import FermionOperator
import numpy as np

Nums = List[int] | List[float] | np.ndarray


@dataclass
class FermionicFragment:
    thetas: Nums
    operators: FermionOperator


@dataclass
class FluidCoeff:
    coeff: Nums
    thetas: Nums


@dataclass
class FluidFermionicFragment(FermionicFragment):
    diag_thetas: Nums
    static_frags: Nums
    fluid_frags: List[FluidCoeff]


@dataclass
class GFROFragment(FermionicFragment):
    lambdas: Nums


@dataclass
class LRFragment(FermionicFragment):
    coeffs: Nums
    diag_coeffs: Nums
    outer_coeff: float


@dataclass
class FragmentedHamiltonian:
    constant: any
    one_body: FermionicFragment
    two_body: List[FermionicFragment]
