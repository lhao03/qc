from typing import Tuple

import numpy as np
from openfermion import FermionOperator


def is_chemist_ordered(term: Tuple) -> bool:
    if len(term) % 2 != 0:
        return False
    creation_operator = True
    for _, op_type in term:
        if not creation_operator and bool(op_type):
            return False
        creation_operator = not creation_operator
    return True


def get_n_body_tensor(fo: FermionOperator, n: int, m: int) -> np.ndarray:
    """Gets the rank 2^n tensor for FermionOperator, where the output tensor will be rank 2^n with dimension m,
     where m is the number of spin orbitals. Assumes the chemist ordering of operators.

    Args:
         n: refers to the `n`-body of the `FermionOperator`
         m: number of orbitals
         fo: an n-body tensor in chemist's ordering

    Raises:
        ValueError: if operator not in chemist ordering

    Returns:
        rank 2^n tensor with dimensions m
    """
    n = 2**n
    dimensions = [m for _ in range(n)]
    tensor = np.zeros(tuple(dimensions), dtype=np.float64)
    for term, coeff in fo.terms.items():
        if not is_chemist_ordered(term):
            raise ValueError(f"Expected chemist ordering, got: {term}")
        indices = [term[i][0] for i in range(n)]
        corr_pos = tensor
        for i in range(n - 1):
            corr_pos = corr_pos[indices[i]]
        corr_pos[indices[-1]] = coeff
    return tensor


def get_n_body_fo(tensor: np.ndarray) -> FermionOperator:
    """Creates the n-body `FermionOperator` from a rank log2(n) tensor"""
    n = tensor.shape[0]
    fo = FermionOperator()
