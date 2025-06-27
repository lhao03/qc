from dataclasses import dataclass
from typing import List, Optional

import numpy as np

Nums = List[int] | List[float] | np.ndarray


@dataclass
class MConfig:
    xpoints: List[float]
    num_spin_orbs: int
    mol_name: str
    mol_of_interest: any
    stable_bond_length: float
    date: str
    folder: Optional[str] = None
