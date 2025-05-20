from typing import Union

from openfermion import (
    QubitOperator,
    FermionOperator,
    normal_ordered,
    reverse_jordan_wigner,
    get_number_preserving_sparse_operator,
)


def get_projected_sparse_op(
    H_OF: Union[FermionOperator, QubitOperator],
    n_qubits,
    num_elecs,
    NSz2SSq_Proj_sparse,
    spin_preserving=True,
    excitation_level=None,
    reference_determinant=None,
):
    # H_OF is a FermionOperator in full space
    if isinstance(H_OF, QubitOperator):
        H_OF = normal_ordered(reverse_jordan_wigner(H_OF))
    first_projected_op = get_number_preserving_sparse_operator(
        H_OF,
        n_qubits,
        num_elecs,
        spin_preserving=spin_preserving,
        excitation_level=excitation_level,
        reference_determinant=reference_determinant,
    )
    if excitation_level is None:
        return NSz2SSq_Proj_sparse * first_projected_op * NSz2SSq_Proj_sparse.T
    else:
        raise UserWarning("Should not reach this branch")


def get_projected_sparse_vec(
    vec, CISD=False
):  # vec must be a M x 1 sparse operator where M is the dimension of N = \eta, Sz = 0 subspace.
    if CISD == False:
        return NSz2SSq_Proj_sparse * vec
    else:
        return NSz2SSq_CISD_Proj_sparse * vec
