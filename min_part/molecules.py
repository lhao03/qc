import os.path
import time

import numpy as np
from openfermion import jordan_wigner, qubit_operator_sparse

from d_types.config_types import MConfig
from min_part.tensor import obt2op
from tests.utils.sim_tensor import get_tensors


def mol_n2(i):
    return [["N", [0, 0, 0]], ["N", [0, 0, i]]]


def mol_h4(i):
    return [
        ["H", [0, 0, 0]],
        ["H", [0, 0, i]],
        ["H", [0, 0, 2 * i]],
        ["H", [0, 0, 3 * i]],
    ]


def mol_h2(i):
    return [["H", [0, 0, 0]], ["H", [0, 0, i]]]


f3_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.f3"
frag_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags"

h2_settings = MConfig(
    xpoints=list(np.linspace(0.2, 3, endpoint=False, num=56)),
    num_spin_orbs=4,  # H2 is 4  # H4 is 4(1s) = 8
    mol_name="H2",
    mol_of_interest=mol_h2,
    stable_bond_length=0.8,
    date=time.strftime("%m-%d-%H%M%S"),
    f3_folder=f3_folder,
    frag_folder=os.path.join(frag_folder, "h2"),
)

h4_settings = MConfig(
    xpoints=list(np.linspace(0.2, 3, endpoint=False, num=56)),
    num_spin_orbs=8,  # H2 is 4  # H4 is 4(1s) = 8
    mol_name="H4",
    mol_of_interest=mol_h4,
    stable_bond_length=0.8,
    date=time.strftime("%m-%d-%H%M%S"),
    f3_folder=f3_folder,
    frag_folder=os.path.join(frag_folder, "h4"),
)
const, obt, tbt = get_tensors(h2_settings, 0.8)
print(qubit_operator_sparse(jordan_wigner(obt2op(obt))).toarray())
