import os
import random
import uuid

import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    FermionOperator,
)
from openfermion.linalg import qubit_operator_sparse

from ham_utils import obtain_OF_hamiltonian
from min_part.ham_decomp import gfro_decomp
from min_part.utils import diag_partitioned_fragments, save_frags
from molecules import mol_h2
from pert_trotter.fermi_frag import Do_GFRO
from plots import plot_energies
from tensor_utils import get_chem_tensors, obt2op, tbt2op
from utils import do_lr_fo

visualize = False
no_partitioning = []

lr_n_subspace_energies = []
lr_all_subspace_energies = []
gfro_n_subspace_energies = []
gfro_all_subspace_energies = []

xpoints = [x * 0.05 for x in list(range(2, 60))]
num_spin_orbs = 4  # H2 is 4  # H4 is 4(1s) = 8
mol_name = "H2"
mol_of_interest = mol_h2
global_id = random.randint(0, 100)

for bond_length in xpoints:
    id = uuid.uuid4()
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
    no_partitioning.append(eigenvalue_0)

    gfro_data = gfro_decomp(tbt=H_tbt)
    gfro_frags = [f.operators for f in gfro_data]
    # _, gfro_frags, _ = Do_GFRO(
    #     H_ele, shrink_frag=False, CISD=False
    # )
    _, _, lr_frags = do_lr_fo(H_ele, projector_func=None)

    H_no_two_body = H_const * FermionOperator.identity() + H_ob_op
    eigenvalues, eigenvectors = sp.linalg.eigh(
        qubit_operator_sparse(jordan_wigner(H_no_two_body)).toarray()
    )

    lr_n_subspace_energy, lr_all_subspace_energy = diag_partitioned_fragments(
        lr_frags, eigenvalues, eigenvectors, num_elecs, num_spin_orbs
    )
    lr_n_subspace_energies.append(lr_n_subspace_energy)
    lr_all_subspace_energies.append(lr_all_subspace_energy)

    gfro_n_subspace_energy, gfro_all_subspace_energy = diag_partitioned_fragments(
        gfro_frags, eigenvalues, eigenvectors, num_elecs, num_spin_orbs
    )
    gfro_n_subspace_energies.append(gfro_n_subspace_energy)
    gfro_all_subspace_energies.append(gfro_all_subspace_energy)

    dir = f"../data/h2/05-28/{global_id}/{id}"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    save_frags(gfro_data, os.path.join(dir, f"gfro_{id}_{mol_name}"))
    save_frags(lr_frags, os.path.join(dir, f"lr_{id}_{mol_name}"))

dir = f"../data/h2/05-28/{global_id}"
plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_subspace_energies,
        gfro_n_subspace_energies,
        lr_all_subspace_energies,
        gfro_all_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, All Bounds, {global_id}",
    labels=[
        "No Partitioning",
        "LR F(M, 2)",
        "GFRO F(M, 2)",
        "LR: All Fock Space",
        "GFRO: All Fock Space",
    ],
    dir=dir
)

plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_subspace_energies,
        gfro_n_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, Tightest Bounds, {global_id}",
    labels=[
        "No Partitioning",
        "LR F(M, 2)",
        "GFRO F(M, 2)",
    ],
    dir=dir,
)
