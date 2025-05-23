import numpy as np
from openfermion import (
    number_operator,
    qubit_operator_sparse,
    jordan_wigner,
    FermionOperator,
)

def fermionic_particle_number_operator(modes: int) -> FermionOperator:
    """Makes an operator that returns number of occupied spin orbitals, in 2nd quantization.

    Formula is: $\hat{N} = \sum_i^M a^\dagger_i a_i$ where M is the number of `modes`, or possible spin orbitals.

    Args:
        modes (int): number of possible spin orbitals

    Returns:
        occupation number operator in form of `FermionOperator`
    """
    fo = FermionOperator('')
    for i in range(modes):
        fo += FermionOperator(((i, 1), (i, 0)))
    return fo

def square_operator(fo: FermionOperator) -> FermionOperator:
    pass

def make_spin_sq_operator(p: int):
    """Makes the S^2 operator.

    Formula is: S^2 = S_x^2 + S_y^2 + S_z^2
    In second quantization is,
    """
    square_operator(make_spin_z_operator(p)) + square_operator(make_spin_x_operator(p)) + square_operator(make_spin_y_operator(p))

def make_spin_x_operator(p: int):
    s_x = FermionOperator('')
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator((even_orb, 1), (odd_orb, 0))
        beta_term = FermionOperator((odd_orb, 1), (even_orb+1, 0))
        s_x += (alpha_term + beta_term)
    return 1/2 * s_x

def make_spin_y_operator(p: int):
    s_y = FermionOperator('')
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator((even_orb, 1), (odd_orb, 0))
        beta_term = FermionOperator((odd_orb, 1), (even_orb, 0))
        s_y += (alpha_term - beta_term)
    return 1/ 2.j * s_y


def make_spin_z_operator(p: int):
    """Makes the S_z operator in 2nd quantization.

    Formula is S_z = 1/2 sum_p (a^\dag_p \alpha a_p \alpha - a^\dag_p \beta a_p \beta).

    In line with openfermion, even numbered operators correspond to alpha spin and
    odd numbered operators correspond to beta spin.

    Args:
        p (int): number of orbitals, N, where N are alpha spin and N are beta spin,
        resulting in total 4 spin-specific orbitals.
    """
    s_z = FermionOperator('')
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator((even_orb, 1), (even_orb, 0))
        beta_term = FermionOperator((odd_orb, 1), (odd_orb, 0))
        s_z += (alpha_term - beta_term)
    return 1/2 * s_z

def get_on_num(w, e: int) -> float:
    """Finds the number of occupied spin orbitals given an eigenvector that was calculated via diagonalization,
    should be a Slater determinant, should be a Slater Determinant.

    Args:
        w (np.array): eigenvector
        e (int): max number of spin orbitals
    Returns:
        number of occupied spin orbitals
    """
    on_op = number_operator(n_modes=e, parity=-1)
    on_op_sparse = qubit_operator_sparse(jordan_wigner(on_op))
    b = on_op_sparse * w
    n = np.divide(b, w)
    n = n[~np.isnan(n)]
    n = n[0]
    return n

def get_spin_sq(w, s_2) -> float:
    pass