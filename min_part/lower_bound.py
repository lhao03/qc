import numpy as np

from proj_ham import get_projected_sparse_op
from utils import EnergyOccupation, do_lr_fo
from ham_utils import obtain_OF_hamiltonian
from tensor_utils import get_chem_tensors, obt2op, tbt2op

from openfermion.linalg import qubit_operator_sparse

from openfermion import (
    count_qubits,
    jordan_wigner,
    s_squared_operator,
    get_number_preserving_sparse_operator,
)

import scipy as sp


from plots import plot_energies
from utils import get_on_num

visualize = False
gs_energies = []
n_subspace_energies = []
all_subspace_energies = []
xpoints = [x * 0.1 for x in list(range(2, 30))]
num_spin_orbs = 4 # H2 is 4  # H4 is 4(1s) = 8
mol_name = "h2-debug"

def mol_n2(i):
    return [["N", [0, 0, 0]], ["N", [0, 0, i]]]


def mol_h4(i):
    return [
        ["H", [0, 0, 0]],
        ["H", [0, 0, i]],
        ["H", [0, 0, 2 * i]],
        ["H", [0, 0, 3 * i]],
    ]

def mol_h2(i):
    return [["H", [0, 0, 0]], ["H", [0, 0, i]]]

mol_of_interest = mol_h2

for bond_length in xpoints:
    print(f"Now simulating: {bond_length} angstroms.")
    mol = mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
    H_ob_op = obt2op(H_obt)
    H_tb_op = tbt2op(H_tbt)
    H_ele = H_const + H_ob_op + H_tb_op
    jw_of = jordan_wigner(H_ele)
    jw_of_sp = qubit_operator_sparse(jw_of)
    jw_op_array = jw_of_sp.toarray()

    eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
    eigenvalue_0 = eigenvalues[0]
    eigenvectors_0 = eigenvectors[:, [0]]
    eigenvectors_0_sparse = sp.sparse.csc_matrix(eigenvectors_0)
    gs_energies.append(eigenvalue_0)

    print("Beginning the Partitioning")
    const, H1, H2_frags = do_lr_fo(H_ele, projector_func=None)
    dfs = []
    allowed_energies = []
    all_energies = []
    for frag in H2_frags:
        jw_of = jordan_wigner(frag)
        jw_of_sp = qubit_operator_sparse(jw_of)
        jw_op_array = jw_of_sp.toarray()
        eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
        tb = []
        energies = []
        all_en = []
        for i in range(eigenvectors.shape[0]):
            energy = eigenvalues[i]
            w = eigenvectors[:, [i]]
            n = get_on_num(w, e=num_spin_orbs)
            tb.append(EnergyOccupation(energy=energy, spin_orbs=n))
            all_en.append(energy)
            if n == num_elecs:
                energies.append(energy)
        all_energies.append(all_en)
        allowed_energies.append(energies)
    H_no_two_body = const + H1
    eigenvalues, eigenvectors = sp.linalg.eigh(qubit_operator_sparse(jordan_wigner(H_no_two_body)).toarray())
    energy_no_two_body = eigenvalues[0]
    eigenvectors_0 = eigenvectors[:, [0]]
    eigenvectors_0_sparse = sp.sparse.csc_matrix(eigenvectors_0)
    two_body_contributions = sum([min(e, default=0) for e in allowed_energies])
    two_body_contributions_not_filtered = sum([min(e, default=0) for e in all_energies])
    print(
        f"LR Energy using only a subset of occupied spin orbital states: {energy_no_two_body + two_body_contributions} Hartree"
    )
    print(
        f"LR Energy using only any spin orbital states: {energy_no_two_body + two_body_contributions_not_filtered} Hartree"
    )
    n_subspace_energies.append(energy_no_two_body + two_body_contributions)
    all_subspace_energies.append(
        energy_no_two_body + two_body_contributions_not_filtered
    )

all_projected_energies = []
for bond_length in xpoints:
    mol = mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
    H_ob_op = obt2op(H_obt)
    H_tb_op = tbt2op(H_tbt)
    H_ele = H_const + H_ob_op + H_tb_op
    jw_of = jordan_wigner(H_ele)
    jw_of_sp = qubit_operator_sparse(jw_of)
    jw_op_array = jw_of_sp.toarray()
    eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)

    s_sq = s_squared_operator(n_qubits // 2)
    s_sq_sparse = get_number_preserving_sparse_operator(
        s_sq, n_qubits, num_elecs, spin_preserving=True
    )
    s_sq_array = s_sq_sparse.toarray()
    s_sq_values, s_sq_vectors = np.linalg.eigh(s_sq_array)
    s_sq_vectors_sparse = sp.sparse.csc_matrix(s_sq_vectors)
    non_cisd_dim = len(list(filter(lambda n: n <= 0.01, s_sq_values)))
    s_sq_evals, nsz2ssq_proj = (
        s_sq_values[:non_cisd_dim],
        s_sq_vectors[:, :non_cisd_dim].T,
    )
    nsz2ssq_proj_sparse = sp.sparse.csc_matrix(nsz2ssq_proj)

    def projector_func(f, excitation_level) -> np.array:
        return get_projected_sparse_op(
            H_OF=f,
            NSz2SSq_Proj_sparse=nsz2ssq_proj_sparse,
            n_qubits=n_qubits,
            num_elecs=num_elecs,
            excitation_level=excitation_level,
        )

    const, H1, H2_frags = do_lr_fo(H_ele, projector_func=projector_func, project=True)
    all_energies = []
    for frag in H2_frags:
        jw_op_array = frag.toarray()
        eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
        all_energies.append(eigenvalues)
    H_no_two_body = const + H1
    jw_op_array = H_no_two_body.toarray()
    eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
    energy_no_two_body = eigenvalues[0]
    two_body_contributions_not_filtered = sum([min(e) for e in all_energies])
    print(
        f"LR Energy using Projection onto GS: {energy_no_two_body + two_body_contributions_not_filtered} Hartree"
    )
    all_projected_energies.append(
        energy_no_two_body + two_body_contributions_not_filtered
    )

plot_energies(
    xpoints=xpoints,
    points=[
        gs_energies,
        n_subspace_energies,
        all_subspace_energies,
    ],
    title=f"{mol_name} Energies, Basis Set STO-3G",
    labels=[
        "No Partitioning",
        "F(M, N) Subspace",
        "All Fock Space",
    ],
)



