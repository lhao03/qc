import os.path
from enum import Enum
from typing import List

import os
import matplotlib.pyplot as plt


class FluidPlotNames(Enum):
    NO_PARTITIONING = "Exact"

    GFRO = "GFRO"
    GFRO_FLUID = "GFRO Fluid"

    LR_N_S = "LR"
    GFRO_N_S = "GFRO"

    LR_F_SPACE = "LR: All Fock Space"
    GFRO_F_SPACE = "GFRO: All Fock Space"

    @classmethod
    def get_color(cls, label):
        match label:
            case RefLBPlotNames.NO_PARTITIONING:
                return ColorBlindFriendly.BLUE
            case RefLBPlotNames.LR_N:
                return ColorBlindFriendly.ORANGE
            case RefLBPlotNames.GFRO_N:
                return ColorBlindFriendly.PINK
            case RefLBPlotNames.LR_N_S:
                return ColorBlindFriendly.BROWN
            case RefLBPlotNames.GFRO_N_S:
                return ColorBlindFriendly.PURPLE
            case RefLBPlotNames.LR_F_SPACE:
                return ColorBlindFriendly.RED
            case RefLBPlotNames.GFRO_F_SPACE:
                return ColorBlindFriendly.YELLOW


class RefLBPlotNames(Enum):
    DIFF = "Difference"
    GFRO = "GFRO"
    F3_GFRO = "Fluid GFRO"

    LR = "LR"
    F3_LR = "Fluid LR"
    NO_PARTITIONING = "Exact"

    @classmethod
    def get_color(cls, label):
        match label:
            case RefLBPlotNames.NO_PARTITIONING:
                return ColorBlindFriendly.BLUE
            case RefLBPlotNames.GFRO:
                return ColorBlindFriendly.ORANGE
            case RefLBPlotNames.F3_GFRO:
                return ColorBlindFriendly.PINK
            case RefLBPlotNames.LR:
                return ColorBlindFriendly.BROWN
            case RefLBPlotNames.F3_LR:
                return ColorBlindFriendly.PURPLE
            case RefLBPlotNames.DIFF:
                return ColorBlindFriendly.RED
            # case RefLBPlotNames.GFRO_F_SPACE:
            #     return ColorBlindFriendly.YELLOW


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
    points: List[List[float]],
    labels: List[RefLBPlotNames | FluidPlotNames],
    title: str,
    dir: str,
):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    plt.clf()
    for label, set_of_points in zip(labels, points):
        plt.plot(
            xpoints,
            set_of_points,
            color=RefLBPlotNames.get_color(label),
            marker="|",
            alpha=0.6,
        )
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Ha)")
    plt.legend([l.value for l in labels])
    plt.title(title)
    plt.savefig(os.path.join(dir, f"{title}.svg"))
