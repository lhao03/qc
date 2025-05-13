from openfermion import (
    QubitOperator,
    FermionOperator,
    normal_ordered,
    reverse_jordan_wigner,
)


def get_projected_sparse_op(
    H_OF: [FermionOperator | QubitOperator],
    n_qubits,
    num_elecs,
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
    elif excitation_level == 2:
        return (
            NSz2SSq_CISD_Proj_sparse * first_projected_op * NSz2SSq_CISD_Proj_sparse.T
        )
    else:
        print("Invalid excitation level. Should be 2 or None.")
        return None


def get_projected_sparse_vec(
    vec, CISD=False
):  # vec must be a M x 1 sparse operator where M is the dimension of N = \eta, Sz = 0 subspace.
    if CISD == False:
        return NSz2SSq_Proj_sparse * vec
    else:
        return NSz2SSq_CISD_Proj_sparse * vec
