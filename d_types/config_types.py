import os.path
from dataclasses import dataclass
from enum import Enum
from time import strftime
from typing import List

import numpy as np

Nums = List[int] | List[float] | np.ndarray


f3_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.f3"
frag_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags"
results_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/data"


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
    f3_folder: str = f3_folder
    frag_folder: str = frag_folder
    results_folder: str = results_folder

    def get_unique_folder(self, folder: str):
        return os.path.join(folder, self.mol_name, self.date)

    def get_unique_report_folder(self):
        return self.get_unique_folder(self.results_folder)

    def get_unique_f3_folder(self):
        return self.get_unique_folder(self.f3_folder)

    def get_unique_frag_folder(self):
        return self.get_unique_folder(self.frag_folder)


class PartitionStrategy(Enum):
    GFRO = "GFRO"
    LR = "LR"


class ContractPattern(Enum):
    GFRO = "r,rp,rq->pq"
    LR = "r,pr,qr->pq"
