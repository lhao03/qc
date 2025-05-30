import os.path
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class PlotNames(Enum):
    NO_PARTITIONING = "No Partitioning"

    LR_N = "LR F(M, 2)"
    GFRO_N = "GFRO F(M, 2)"

    LR_N_S = "LR F(M, 2) + Spin"
    GFRO_N_S = "GFRO F(M, 2) + Spin"

    LR_F_SPACE = "LR: All Fock Space"
    GFRO_F_SPACE = "GFRO: All Fock Space"

    @classmethod
    def get_color(cls, label):
        match label:
            case PlotNames.NO_PARTITIONING:
                return ColorBlindFriendly.BLUE
            case PlotNames.LR_N:
                return ColorBlindFriendly.ORANGE
            case PlotNames.GFRO_N:
                return ColorBlindFriendly.PINK
            case PlotNames.LR_N_S:
                return ColorBlindFriendly.BROWN
            case PlotNames.GFRO_N_S:
                return ColorBlindFriendly.PURPLE
            case PlotNames.LR_F_SPACE:
                return ColorBlindFriendly.RED
            case PlotNames.GFRO_F_SPACE:
                return ColorBlindFriendly.YELLOW


class ColorBlindFriendly:
    BLUE = "#377eb8"
    ORANGE = "#ff7f00"
    PINK = "#f781bf"
    BROWN = "#a65628"
    PURPLE = "#984ea3"
    RED = "#e41a1c"
    YELLOW = "#FFC300"


def plot_energies(
    xpoints: List[float],
    points: List[np.ndarray],
    labels: List[PlotNames],
    title: str,
    dir: str,
):
    plt.clf()
    for label, set_of_points in zip(labels, points):
        plt.plot(xpoints, set_of_points, color=PlotNames.get_color(label))
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Ha)")
    plt.legend([l.value for l in labels])
    plt.title(title)
    plt.savefig(os.path.join(dir, f"{title}.svg"))
