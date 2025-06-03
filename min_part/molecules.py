import numpy as np

from min_part.utils import LowerBoundConfig


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


h2_settings = LowerBoundConfig(
    xpoints=list(np.arange(0.2, 3, 0.05)),
    num_spin_orbs=4,  # H2 is 4  # H4 is 4(1s) = 8
    mol_name="H2",
    mol_of_interest=mol_h2,
    stable_bond_length=0.8,
)
