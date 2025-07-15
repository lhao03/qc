from dataclasses import dataclass
from enum import Enum
from time import strftime
from typing import List, Optional

import numpy as np

Nums = List[int] | List[float] | np.ndarray


@dataclass
class MConfig:
    xpoints: List[float]
    num_spin_orbs: int
    gs_elecs: int
    s2: int
    sz: int
    mol_name: str
    mol_coords: any
    stable_bond_length: float
    date: str = strftime("%m-%d-%H%M%S")
    f3_folder: Optional[str] = None
    frag_folder: Optional[str] = None


class PartitionStrategy(Enum):
    GFRO = "GFRO"
    LR = "LR"


class ContractPattern(Enum):
    GFRO = "r,rp,rq->pq"
    LR = "r,pr,qr->pq"
