#%% md
# # Part 1: Generate Hamiltonian, and Partition it
#%%
from pert_trotter.ham_utils import obtain_OF_hamiltonian

from openfermion import (
    count_qubits,
    jordan_wigner,
)

from pert_trotter.tensor_utils import get_chem_tensors, obt2op, tbt2op

mol = [["H", [0, 0, 0]], ["H", [0, 0, 0.8]]]
H, num_elecs = obtain_OF_hamiltonian(mol)
n_qubits = count_qubits(H)
assert n_qubits == 4
#%%
from openfermion import jw_number_restrict_operator
from openfermion.linalg import qubit_operator_sparse

H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
H_ob_op = obt2op(H_obt)
H_tb_op = tbt2op(H_tbt)
H_ele = H_const + H_ob_op + H_tb_op
print(H_obt.shape)
print(H_tbt.shape)
jw_of = jordan_wigner(H_ele)
jw_of_sp = qubit_operator_sparse(jw_of)
print(jw_of_sp.shape)
print(jw_of_sp)
#%%
# generate the exact eigenstates
import scipy as sp
jw_op_array = jw_of_sp.toarray()
eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
eigenvalue_0 = eigenvalues[0]
eigenvectors_0 = eigenvectors[:, [0]]
eigenvectors_0_sparse = sp.sparse.csc_matrix(eigenvectors_0)
print(eigenvalue_0)
print(eigenvectors)
#%%
from openfermion import jw_slater_determinant

# prepare n-electron slater determinants with N spin-orbitals

jw_slater_determinant()
#%%
from pert_trotter.fermi_frag import Do_Fermi_Partitioning

def sum_partitions(frags, num_e):
    min_e_each_frag = []
    for i, frag in enumerate(frags):
        print(frags.shape)
        frag_n = jw_number_restrict_operator(frag, n_electrons=num_e)
        values, wectors = sp.linalg.eigh(frag_n.toarray())
        min_e_each_frag.append(min(values))
    return sum(min_e_each_frag)

fragment_array_sarray = Do_Fermi_Partitioning(
            H_ele,
            type="lr",
            tol=1e-6,
            spacial=False,
            save=False,
            load=False,
        )

summed_energies = sum_partitions(frags=fragment_array_sarray, num_e=2)
print(summed_energies)
#%%
import os
from pathlib import Path

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    qubit_operator_sparse,
    s_squared_operator,
    get_number_preserving_sparse_operator,
    get_sparse_operator,
    get_ground_state,
    jw_slater_determinant,
)

from pert_trotter import config
from pert_trotter.fermi_frag import Do_Fermi_Partitioning
from pert_trotter.proj_ham import get_projected_sparse_op
from pert_trotter.tensor_utils import get_chem_tensors, obt2op, tbt2op
from plots import plot_energies

methods = ["lr", "gfro", "lrlcu", "gfrolcu", "sdgfro"]

def obtain_slater_determinant(num_elecs, num_orbitals):
    """
    Generate a Slater Determinant based on number of electrons and spin orbitals.
    """
    q = 1/np.sqrt(2.0) * np.array([[1, 1]])
    return np.array(jw_slater_determinant(q))

print(obtain_slater_determinant(num_elecs=1,num_orbitals=1))

def get_frag_energies_projected(mol_name, mol):

    # build projector
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
    # hamiltonian projection, lr
    method = methods[0]
    method_path = f"data/{method}/sparse_arrays/full_space/"
    full_path = os.path.join(os.getcwd(), method_path)
    Path(full_path).mkdir(parents=True, exist_ok=True)
    fragment_array_sarray = Do_Fermi_Partitioning(
        H_ele,
        type=method,
        tol=1e-6,
        spacial=False,
        save=False,
        load=False,
        shrink_frag=True,
        projector_func=lambda f, excitation_level: get_projected_sparse_op(
            H_OF=f,
            NSz2SSq_Proj_sparse=nsz2ssq_proj_sparse,
            n_qubits=n_qubits,
            num_elecs=num_elecs,
            excitation_level=excitation_level,
        ),
    )
    summed_energies = sum_frags(fragment_array_sarray)
    return gs_energy, summed_energies

def get_frag_energies(mol_name, mol):
        # make the molecular hamiltonian
        H, num_elecs = obtain_OF_hamiltonian(mol)
        n_qubits = count_qubits(H)  # 1s 1s
        H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
        H_ob_op = obt2op(H_obt)
        H_tb_op = tbt2op(H_tbt)
        H_ele = H_const + H_ob_op + H_tb_op
        jw_op = jordan_wigner(H_ele)
        gs_energy = get_ground_state(get_sparse_operator(jw_op))[0]
        config.mol_name = mol_name
        config.n_qubits = n_qubits
        config.num_elecs = num_elecs
        # generate the exact eigenstates
        jw_op_array_sparse = qubit_operator_sparse(jw_op)
        jw_op_array = jw_op_array_sparse.toarray()
        eigenvalues, eigenvectors = sp.linalg.eigh(jw_op_array)
        eigenvalue_0 = eigenvalues[0]
        eigenvectors_0 = eigenvectors[:, [0]]
        eigenvectors_0_sparse = sp.sparse.csc_matrix(eigenvectors_0)
        # hamiltonian projection, lr
        method = methods[0]
        fragment_array_sarray = Do_Fermi_Partitioning(
            H_ele,
            type=method,
            tol=1e-6,
            spacial=False,
            save=False,
            load=False,
        )
        summed_energies = sum_frags(fragment_array_sarray)
        return gs_energy, summed_energies

def h2_projection():
    gs_energies = []
    min_energies = []
    xpoints =  np.linspace(0.2, 3, 20)
    for i in xpoints:
        gs, min_e = get_frag_energies_projected(
            mol_name="h2",
            mol=[["H", [0, 0, 0]], ["H", [0, 0, i]]],
        )
        gs_energies.append(gs)
        min_energies.append(min_e)
    plot_energies(xpoints, gs_energies, min_energies, "H2 with Projection")

def h2_no_projection():
    gs_energies = []
    min_energies = []
    xpoints =  np.linspace(0.2, 3, 20)
    for i in xpoints:
        gs, min_e = get_frag_energies(
            mol_name="h2",
            mol=[["H", [0, 0, 0]], ["H", [0, 0, i]]],
        )
        gs_energies.append(gs)
        min_energies.append(min_e)
    plot_energies(xpoints, gs_energies, min_energies, "H2 No Projection")

# def test_lih_projection():
#     gs_energies = []
#     min_energies = []
#     xpoints =  np.linspace(1, 2, 10)
#     for i in xpoints:
#         gs, min_e = get_frag_energies_projected(
#             mol_name="lih",
#             mol=[["Li", [0, 0, 0]], ["H", [0, 0, i]]],
#         )
#         gs_energies.append(gs)
#         min_energies.append(min_e)
#     plot_energies(xpoints, gs_energies, min_energies, "LiH with Projection")

# def test_lih_no_projection():
#     gs_energies = []
#     min_energies = []
#     xpoints =  np.linspace(1, 2, 10)
#     for i in xpoints:
#         gs, min_e = get_frag_energies(
#             mol_name="lih",
#             mol=[["Li", [0, 0, 0]], ["H", [0, 0, i]]],
#         )
#         gs_energies.append(gs)
#         min_energies.append(min_e)
#     plot_energies(xpoints, gs_energies, min_energies, "LiH No Projection")

def h4():
    gs_energies = []
    min_energies = []
    xpoints =  np.linspace(0.2, 1.5, 20)
    for i in xpoints:
        gs, min_e = get_frag_energies_projected(
            mol_name="h4",
            mol= [
            ["H", [0, 0, 0]],
            ["H", [0, 0, i]],
            ["H", [0, 0, 2]],
            ["H", [0, 0, 3]],
        ],
        )
        gs_energies.append(gs)
        min_energies.append(min_e)
    plot_energies(xpoints, gs_energies, min_energies, "H4 with Projection")