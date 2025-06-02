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


def get_no_from_tensor(lambda_m: np.ndarray) -> FermionOperator:
    """Creates the GFRO fragment in number operator form without orbital rotations.

    sum_{lm} lambda_{lm} n_l n_m

    Args:
        lambda_m: the n by n matrix defining the eigenvalues for the new orbitals from the U matrix. There are m unique values
        in lambda_m, where n * (n + 1) / 2 = m. n are the original number of spin orbitals (usually atomic ones).

    Returns:
        `FermionOperator` form of the fragment: sum_{lm} lambda_{lm} n_l n_m
    """
    s = lambda_m.shape[0]
    gfro_operator = FermionOperator()
    for l in range(s):
        for m in range(s):
            n_l = f"{str(l)}^ {str(l)}"
            n_m = f"{str(m)}^ {str(m)}"
            gfro_operator += FermionOperator(
                term=f"{n_l} {n_m}", coefficient=lambda_m[l][m]
            )
    return gfro_operator
