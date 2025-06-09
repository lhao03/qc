from dataclasses import dataclass
from typing import List


@dataclass
class LowerBoundConfig:
    xpoints: List[float]
    num_spin_orbs: int
    mol_name: str
    mol_of_interest: any
    stable_bond_length: float
    date: str
