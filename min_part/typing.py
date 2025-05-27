from dataclasses import dataclass
from typing import TypeVar, Generic, Tuple, Union, Optional
import numpy as np
from openfermion import FermionOperator

Shape = TypeVar("Shape")
DType = TypeVar("DType")


@dataclass
class GFROFragment:
    lambdas: np.ndarray
    thetas: np.ndarray
    operators: FermionOperator


class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    pass


def compute_l2_norm(arr: Array["N,2", float]) -> Array["N", float]:
    return (arr**2).sum(axis=1) ** 0.5


print(compute_l2_norm(arr=np.array([(1, 2), (3, 1.5), (0, 5.5)])))
