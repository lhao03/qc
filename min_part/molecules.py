import os.path
import time

import numpy as np

from d_types.config_types import MConfig


def mol_n2(i):
    return [["N", [0, 0, 0]], ["N", [0, 0, i]]]


def mol_h4(i):
    return [
        ["H", [0, 0, 0]],
        ["H", [0, 0, i]],
        ["H", [0, 0, 2 * i]],
        ["H", [0, 0, 3 * i]],
    ]


def mol_hh_hh(i):
    return [
        ["H", [0, 0, 0]],
        ["H", [0, 0, i]],
        ["H", [i, 0, i]],
        ["H", [i, 0, 0]],
    ]


def mol_h2(i):
    return [["H", [0, 0, 0]], ["H", [0, 0, i]]]


f3_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.f3"
frag_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags"


def make_points(start, stop, do_rounding: bool = True):
    if do_rounding:
        return list(
            round(f, 2) for f in np.linspace(start, stop, endpoint=False, num=56)
        )
    else:
        return list(np.linspace(start, stop, endpoint=False, num=56))


h2_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=4,  # H2 is 4  # H4 is 4(1s) = 8
    gs_elecs=2,
    s2=0,
    sz=0,
    mol_name="H2",
    mol_coords=mol_h2,
    stable_bond_length=0.8,
    f3_folder=f3_folder,
    frag_folder=os.path.join(frag_folder, "h2"),
)


h4_settings = MConfig(
    xpoints=make_points(0.2, 3, do_rounding=False),
    num_spin_orbs=8,  # H2 is 4  # H4 is 4(1s) = 8
    gs_elecs=4,
    s2=0,
    sz=0,
    mol_name="H4",
    mol_coords=mol_h4,
    stable_bond_length=0.8,
    date=time.strftime("%m-%d-%H%M%S"),
    f3_folder=f3_folder,
    frag_folder=os.path.join(frag_folder, "h4"),
)
