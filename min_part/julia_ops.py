from typing import Tuple, List

import numpy as np
from julia import Pkg, Main

Pkg.activate("../MolHamLinAlg")
from julia import MolHamLinAlg


def extract_eigen(operator, w, panic):
    return MolHamLinAlg.extract_eigen(operator.toarray(), w, panic)


def solve_quad(a, b, c):
    return MolHamLinAlg.solve_quad(a, b, c)


def UV_eigendecomp(matrix) -> Tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.UV_eigendecomp(matrix)


def lr_decomp_params(tbt) -> Tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.lr_decomposition(tbt)
