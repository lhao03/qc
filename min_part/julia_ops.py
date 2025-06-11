from typing import Tuple

import numpy as np
from julia import Pkg

Pkg.activate("../MolHamLinAlg")
from julia import MolHamLinAlg

def extract_eigen(operator, w, panic):
    return MolHamLinAlg.extract_eigen(operator.toarray(), w, panic)

def solve_quad(a, b, c):
    return MolHamLinAlg.solve_quad(a, b, c)

def eigendecomp(matrix) -> Tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.eigendecomp(matrix)