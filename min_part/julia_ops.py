from typing import Tuple, List

import numpy as np
from julia import Pkg, Main

Pkg.activate("../MolHamLinAlg")
from julia import MolHamLinAlg


def check_lr_decomp(tbt, lr_tensors):
    MolHamLinAlg.check_lr_decomp(tbt, lr_tensors)


def jl_print(m):
    MolHamLinAlg.jl_print(m)


def extract_eigen(operator, w, panic):
    return MolHamLinAlg.extract_eigen(operator.toarray(), w, panic)


def solve_quad(a, b, c):
    return MolHamLinAlg.solve_quad(a, b, c)


def eigen_jl(matrix) -> Tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.UV_eigendecomp(matrix)


def rowwise_reshape(A, n) -> np.ndarray:
    return MolHamLinAlg.rowwise_reshape(A, n)


def reshape_eigs(d) -> np.ndarray:
    return MolHamLinAlg.reshape_eigs(d)


def vecs2mat_reshape(L, n) -> List[np.ndarray]:
    return [np.array(a) for a in MolHamLinAlg.vecs2mat_reshape(L, n)]


def lr_decomp_params(tbt) -> Tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.lr_decomposition(tbt)


def jl_compare_matrices(a, b):
    MolHamLinAlg.jl_compare_matrices(a, b)
