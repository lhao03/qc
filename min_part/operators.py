import numpy as np
from openfermion import (
    number_operator,
    qubit_operator_sparse,
    jordan_wigner,
    FermionOperator,
)


def extract_eigenvalue(operator, w):
    b = operator * w
    n = np.divide(b, w)
    n = n[~np.isnan(n)]
    n = n[0]
    return n


def tuple2str(*args) -> str:
    fo_str = []
    for a in args:
        indice = a[0][0]
        type_a = a[0][1]
        fo_str.append(f"{indice}" + ("^" if type_a == 1 else ""))
    return " ".join(fo_str)


def fermionic_particle_number_operator(modes: int) -> FermionOperator:
    """Makes an operator that returns number of occupied spin orbitals, in 2nd quantization.

    Formula is: $\hat{N} = \sum_i^M a^\dagger_i a_i$ where M is the number of `modes`, or possible spin orbitals.

    Args:
        modes (int): number of possible spin orbitals

    Returns:
        occupation number operator in form of `FermionOperator`
    """
    fo = FermionOperator()
    for i in range(modes):
        fo += FermionOperator(((i, 1), (i, 0)))
    return fo


def get_squared_operator(fo: FermionOperator) -> FermionOperator:
    sqed_operator = FermionOperator()
    for term_1, coeff_1 in fo.terms.items():
        for term_2, coeff_2 in fo.terms.items():
            new_term = tuple2str(term_1, term_2)
            new_fo_term = FermionOperator(term=new_term, coefficient=coeff_1 * coeff_2)
            sqed_operator += new_fo_term
    return sqed_operator


def make_shift_up_operator(p: int) -> FermionOperator:
    s_plus = FermionOperator()
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_beta = FermionOperator(((even_orb, 1), (odd_orb, 0)))
        s_plus += alpha_beta
    return s_plus


def make_shift_down_operator(p: int) -> FermionOperator:
    s_minus = FermionOperator()
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        beta_alpha = FermionOperator(((odd_orb, 1), (even_orb, 0)))
        s_minus += beta_alpha
    return s_minus


def make_total_spin_operator(
    p: int,
) -> FermionOperator:
    """Makes the S^2 operator.

    Formula is: S^2 = S_x^2 + S_y^2 + S_z^2 (doesn't work?)
    or S^2 = S_-  S_+ + S_z(S_z + 1) (works, matches OpenFermion's implemetnation)
    """
    s_z = make_spin_z_operator(p)
    shift_plus = make_shift_up_operator(p)
    shift_minus = make_shift_down_operator(p)
    return shift_minus * shift_plus + s_z * (s_z + 1 * FermionOperator.identity())


def make_spin_x_operator(p: int) -> FermionOperator:
    s_x = FermionOperator()
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator(((even_orb, 1), (odd_orb, 0)))
        beta_term = FermionOperator(((odd_orb, 1), (even_orb, 0)))
        s_x += 1 / 2 * (alpha_term + beta_term)
    return s_x


def make_spin_y_operator(p: int) -> FermionOperator:
    s_y = FermionOperator()
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator(((even_orb, 1), (odd_orb, 0)))
        beta_term = FermionOperator(((odd_orb, 1), (even_orb, 0)))
        s_y += 1 / 2.0j * (alpha_term - beta_term)
    return s_y


def make_spin_z_operator(p: int) -> FermionOperator:
    """Makes the S_z operator in 2nd quantization.

    Formula is S_z = 1/2 sum_p (a^\dag_p \alpha a_p \alpha - a^\dag_p \beta a_p \beta).

    In line with openfermion, even numbered operators correspond to alpha spin and
    odd numbered operators correspond to beta spin.

    Args:
        p (int): number of orbitals, N, where N are alpha spin and N are beta spin,
        resulting in total 4 spin-specific orbitals.

    Returns:
        the S_z operator as a  `FermionOperator`
    """
    s_z = FermionOperator()
    for i in range(p):
        even_orb = i * 2
        odd_orb = even_orb + 1
        alpha_term = FermionOperator(((even_orb, 1), (even_orb, 0)))
        beta_term = FermionOperator(((odd_orb, 1), (odd_orb, 0)))
        s_z += 1 / 2 * (alpha_term - beta_term)
    return s_z


def get_particle_number(w, e: int) -> float:
    """Finds the number of occupied spin orbitals given an eigenvector that was calculated via diagonalization,
    should be a Slater determinant, should be a Slater Determinant.

    Args:
        w (np.array): eigenvector that corresponds to a Slater Determinant.
        e (int): total number of spin orbitals
    Returns:
        number of occupied spin orbitals
    """
    on_operator = qubit_operator_sparse(
        jordan_wigner(number_operator(n_modes=e, parity=-1))
    )
    return extract_eigenvalue(on_operator, w)


def get_total_spin(w, p: int) -> float:
    """Finds the total spin of a molecular system. Note in general the Slater determinant is not an eigenfunction
    of the S^2 operator unless it is in a closed-shell state with all paired electrons or high-energy state with
    electrons of all one spin type (ex: all alpha)

    Args:
        w: eigenvector, expecting a Slater determinant
        p (int): number of orbitals, N, where N are alpha spin and N are beta spin,
        resulting in total 4 spin-specific orbitals.

    Returns:
        the projected spin
    """
    s_2 = qubit_operator_sparse(jordan_wigner(make_total_spin_operator(p)))
    return extract_eigenvalue(s_2, w)


def get_projected_spin(w, p: int) -> float:
    """Finds the projected spin of a singlet system, assuming a Slater Determinant is used as the approximated wavefunction.

    Args:
        w: eigenvector, that corresponds to Slater Determinant
        p (int): number of orbitals, N, where N are alpha spin and N are beta spin,
        resulting in total 4 spin-specific orbitals.

    Returns:
        The projected spin, 0 if singlet, 1 if triplet.
    """
    s_z_operator = qubit_operator_sparse(jordan_wigner(make_spin_z_operator(p=p)))
    return extract_eigenvalue(s_z_operator, w)
