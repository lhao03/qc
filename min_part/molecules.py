import time

import numpy as np
from math import radians, cos, sin

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


thetaH2O = radians(107.6 / 2)
xH2O = sin(thetaH2O)
yH2O = cos(thetaH2O)


def mol_h2o(i):
    return [
        ["O", [0, 0, 0]],
        ["H", [i * -xH2O, i * yH2O, 0]],
        ["H", [i * xH2O, i * yH2O, 0]],
    ]


def mol_lih(i):
    return [["Li", [0, 0, 0]], ["H", [0, 0, i]]]


def make_points(start, stop, do_rounding: bool = True):
    if do_rounding:
        return list(
            round(f, 2) for f in np.linspace(start, stop, endpoint=False, num=56)
        )
    else:
        return list(np.linspace(start, stop, endpoint=False, num=56))


lih_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=4,
    gs_elecs=2,
    s2=0,
    sz=0,
    mol_name="LiH",
    mol_coords=mol_lih,
    stable_bond_length=0.8,
)

h2o_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=4 + 2 + 2 + 6,
    gs_elecs=2 + 8,
    s2=0,
    sz=0,
    mol_name="H2O",
    mol_coords=mol_h2o,
    stable_bond_length=0.95,
)

h2_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=4,
    gs_elecs=2,
    s2=0,
    sz=0,
    mol_name="H2",
    mol_coords=mol_h2,
    stable_bond_length=0.8,
)


h4_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=8,
    gs_elecs=4,
    s2=0,
    sz=0,
    mol_name="H4",
    mol_coords=mol_h4,
    stable_bond_length=0.8,
    date=time.strftime("%m-%d-%H%M%S"),
)

h4_sq_settings = MConfig(
    xpoints=make_points(0.2, 3),
    num_spin_orbs=8,
    gs_elecs=4,
    s2=0,
    sz=0,
    mol_name="H4 Square",
    mol_coords=mol_hh_hh,
    stable_bond_length=0.8,
    date=time.strftime("%m-%d-%H%M%S"),
)
