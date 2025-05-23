from math import isclose

import numpy as np

from min_part.operators import get_particle_number, get_total_spin, get_projected_spin
from min_part.utils import choose_lowest_energy
from molecules import mol_h2
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

visualize = False
gs_energies = []
gs_subspace_energies = []
n_subspace_energies = []
all_subspace_energies = []
xpoints = [x * 0.1 for x in list(range(2, 30))]
num_spin_orbs = 4  # H2 is 4  # H4 is 4(1s) = 8
mol_name = "H2"


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
    allowd_energies_more_constraints = []
    all_energies = []
    for frag in H2_frags:
        jw_of = jordan_wigner(frag)
        jw_of_sp = qubit_operator_sparse(jw_of)
        jw_op_array = jw_of_sp.toarray()
        eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
        tb = []
        energies = []
        more_constraints_energies = []
        all_en = []
        for i in range(eigenvectors.shape[0]):
            energy = eigenvalues[i]
            w = eigenvectors[:, [i]]
            n = get_particle_number(w, e=num_spin_orbs)
            s_2 = get_total_spin(w, num_spin_orbs // 2)
            s_z = get_projected_spin(w, num_spin_orbs // 2)
            tb.append(EnergyOccupation(energy=energy, spin_orbs=n))
            all_en.append(energy)
            if n == num_elecs:
                energies.append(energy)
                if isclose(s_2, 0, abs_tol=1e-6) and isclose(s_z, 0, abs_tol=1e-6):
                    more_constraints_energies.append(energy)
        allowd_energies_more_constraints.append(more_constraints_energies)
        all_energies.append(all_en)
        allowed_energies.append(energies)
    H_no_two_body = const + H1
    eigenvalues, eigenvectors = sp.linalg.eigh(
        qubit_operator_sparse(jordan_wigner(H_no_two_body)).toarray()
    )
    energy_no_two_body = choose_lowest_energy(
        eigenvalues, eigenvectors, num_spin_orbs, num_elecs, proj_spin=0, total_spin=0
    )
    energy_no_two_body_no_constraints = min(eigenvalues)
    two_body_contributions_most_constraints = sum(
        [min(e, default=0) for e in allowd_energies_more_constraints]
    )
    two_body_contributions = sum([min(e, default=0) for e in allowed_energies])
    two_body_contributions_not_filtered = sum([min(e, default=0) for e in all_energies])
    energy_gs_sym = energy_no_two_body + two_body_contributions_most_constraints
    energy_only_elecs = energy_no_two_body + two_body_contributions
    energy_not_filtered = energy_no_two_body_no_constraints + two_body_contributions_not_filtered
    print(f"Energy no two body contributions:{energy_no_two_body}")
    print(f"tbe:{two_body_contributions_most_constraints}")
    print(f"LR Energy more constraints: {energy_gs_sym} Hartree")
    gs_subspace_energies.append(energy_gs_sym)
    n_subspace_energies.append(energy_only_elecs)
    all_subspace_energies.append(energy_not_filtered)

all_projected_energies = []
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
    tbe = sum([min(e) for e in all_energies])
    te = energy_no_two_body + tbe
    print(f"Energy no two body contributions:{energy_no_two_body}")
    print(f"tbe:{tbe}")
    print(f"LR Energy using Projection onto GS: {te} Hartree")
    all_projected_energies.append(te)

plot_energies(
    xpoints=xpoints,
    points=[
        gs_energies,
        n_subspace_energies,
        gs_subspace_energies,
        # all_subspace_energies,
        all_projected_energies,
    ],
    title=f"{mol_name} Energies via LR Partitioning, Basis Set STO-3G",
    labels=[
        "No Partitioning",
        "N=2 Ver 1",
        "N=2 Ver 2",
        # "All Fock Space",
        "PIGSoH",
    ],
)
