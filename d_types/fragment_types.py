from dataclasses import dataclass
from typing import List

from openfermion import FermionOperator
import numpy as np


@dataclass
class FermionicFragment:
    thetas: np.ndarray
    operators: FermionOperator


@dataclass
class FluidFermionicFragment(FermionicFragment):
    lambdas: np.ndarray


@dataclass
class GFROFragment(FermionicFragment):
    lambdas: np.ndarray


@dataclass
class LRFragment(FermionicFragment):
    h_p: np.ndarray


@dataclass
class FragmentedHamiltonian:
    constant: any
    one_body: FermionicFragment
    two_body: List[FermionicFragment]
