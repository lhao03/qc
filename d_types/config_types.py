from dataclasses import dataclass
from enum import Enum
from functools import partial
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
    f3_folder: Optional[str] = None
    frag_folder: Optional[str] = None


class PartitionStrategy(Enum):
    GFRO = "GFRO"
    LR = "LR"


class ContractPattern(Enum):
    GFRO = "r,rp,rq->pq"
    LR = "r,pr,qr->pq"


@dataclass
class Subspace:
    expected_e: int
    expected_s2: int
    expected_sz: int
    projector: Optional[partial] = None

    def __eq__(self, other):
        if isinstance(other, Subspace):
            return (
                self.expected_e == other.expected_e
                and self.expected_s2 == other.expected_s2
                and self.expected_sz == other.expected_sz
            )
        else:
            return False
