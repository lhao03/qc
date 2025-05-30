import json
import os
import random
import uuid

import numpy as np
import scipy as sp
from openfermion import (
    count_qubits,
    jordan_wigner,
    FermionOperator,
)
from openfermion.linalg import qubit_operator_sparse

from ham_utils import obtain_OF_hamiltonian
from min_part.ham_decomp import gfro_decomp
from min_part.utils import (
    diag_partitioned_fragments,
    save_frags,
    get_saved_file_names,
    open_frags,
    range_float,
)
from molecules import mol_h2, mol_h4
from pert_trotter.fermi_frag import Do_GFRO
from plots import plot_energies
from tensor_utils import get_chem_tensors, obt2op, tbt2op
from utils import do_lr_fo

visualize = False
no_partitioning = []

lr_n_subspace_energies = []
lr_n_s_subspace_energies = []
lr_all_subspace_energies = []

gfro_n_subspace_energies = []
gfro_n_s_subspace_energies = []
gfro_all_subspace_energies = []
lr_stats = []
gfro_stats = []

xpoints = np.arange(0.2, 3, 0.05)
num_spin_orbs = 4  # H2 is 4  # H4 is 4(1s) = 8
mol_name = "H2"
mol_of_interest = mol_h2
global_id = random.randint(0, 100)
parent_dir = f"../data/{mol_name.lower()}"
child_dir = os.path.join(parent_dir, "05-29", str(global_id))

load = False
if load:
    child_dir = "../data/h4/05-29/11"
    gfro_name = "gfro_<built-in function id>_H4"
    lr_name = "lr_<built-in function id>_H4"
    gfro_files, lr_files = get_saved_file_names(child_dir, gfro_name, lr_name)

for i, bond_length in enumerate(xpoints):
    print(f"Now partitioning: {bond_length} angstroms.")
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

    if load:
        gfro_data = open_frags(gfro_files[i])
        lr_frags = open_frags(lr_files[i])
    else:
        print("Begin GFRO Decomp")
        gfro_data = gfro_decomp(tbt=H_tbt)
        # gfro_frags = Do_GFRO(H_ele, shrink_frag=False, CISD=None)
        print("Begin LR Decomp")
        _, _, lr_frags = do_lr_fo(H_ele, projector_func=None)
    gfro_frags = [f.operators for f in gfro_data]

    H_no_two_body = H_const * FermionOperator.identity() + H_ob_op
    h1_v, h1_w = sp.linalg.eigh(
        qubit_operator_sparse(jordan_wigner(H_no_two_body)).toarray()
    )

    lr_n_subspace_energy, lr_n_s_energy, lr_all_subspace_energy = (
        diag_partitioned_fragments(
            h2_frags=lr_frags,
            h1_v=h1_v,
            h1_w=h1_w,
            num_elecs=num_elecs,
            num_spin_orbs=num_spin_orbs,
        )
    )
    lr_n_subspace_energies.append(lr_n_subspace_energy)
    lr_n_s_subspace_energies.append(lr_n_s_energy)
    lr_all_subspace_energies.append(lr_all_subspace_energy)

    gfro_n_subspace_energy, gfro_n_s_subspace_energy, gfro_all_subspace_energy = (
        diag_partitioned_fragments(
            h2_frags=gfro_frags,
            h1_v=h1_v,
            h1_w=h1_w,
            num_elecs=num_elecs,
            num_spin_orbs=num_spin_orbs,
        )
    )
    gfro_n_subspace_energies.append(gfro_n_subspace_energy)
    gfro_n_s_subspace_energies.append(gfro_n_s_subspace_energy)
    gfro_all_subspace_energies.append(gfro_all_subspace_energy)

    if not load:
        new_dir = os.path.join(child_dir, f"{str(i)}_", str(uuid.uuid4()))
        while os.path.exists(new_dir):
            new_dir = os.path.join(child_dir, str(uuid.uuid4()))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        save_frags(gfro_data, os.path.join(new_dir, f"{str(i)}_gfro_{mol_name}"))
        save_frags(lr_frags, os.path.join(new_dir, f"{str(i)}_lr_{mol_name}"))
    print(
        f"GFRO Energy Difference: {eigenvalue_0 - gfro_n_subspace_energy}, GFRO + Spin Energy Difference: {eigenvalue_0 - gfro_n_s_subspace_energy}"
    )
    print(
        f"LR Energy Difference: {eigenvalue_0 - lr_n_subspace_energy}, GFRO + Spin Energy Difference: {eigenvalue_0 - lr_n_s_energy}"
    )

plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_subspace_energies,
        gfro_n_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, No Spin Constraints",
    labels=[
        "No Partitioning",
        "LR F(M, 2)",
        "GFRO F(M, 2)",
    ],
    dir=child_dir,
)

plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_s_subspace_energies,
        gfro_n_s_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, Spin Constraints",
    labels=[
        "No Partitioning",
        "LR F(M, 2) + Spin",
        "GFRO F(M, 2) + Spin",
    ],
    dir=child_dir,
)

plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_subspace_energies,
        gfro_n_subspace_energies,
        lr_n_s_subspace_energies,
        gfro_n_s_subspace_energies,
        lr_all_subspace_energies,
        gfro_all_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, All Bounds",
    labels=[
        "No Partitioning",
        "LR F(M, 2)",
        "GFRO F(M, 2)",
        "LR F(M, 2) + Spin",
        "GFRO F(M, 2) + Spin",
        "LR: All Fock Space",
        "GFRO: All Fock Space",
    ],
    dir=child_dir,
)

plot_energies(
    xpoints=xpoints,
    points=[
        no_partitioning,
        lr_n_subspace_energies,
        gfro_n_subspace_energies,
        lr_n_s_subspace_energies,
        gfro_n_s_subspace_energies,
    ],
    title=f"{mol_name} Energies from Partitioning, Tightest Bounds",
    labels=[
        "No Partitioning",
        "LR F(M, 2)",
        "GFRO F(M, 2)",
        "LR F(M, 2) + Spin",
        "GFRO F(M, 2) + Spin",
    ],
    dir=child_dir,
)

energies = {
    "no_partitioning": no_partitioning,
    "lr_n_subspace_energies": lr_n_subspace_energies,
    "gfro_n_subspace_energies": gfro_n_subspace_energies,
    "lr_n_s_subspace_energies": lr_n_s_subspace_energies,
    "gfro_n_s_subspace_energies": gfro_n_s_subspace_energies,
    "lr_all_subspace_energies": lr_all_subspace_energies,
    "gfro_all_subspace_energies": gfro_all_subspace_energies,
}

energies_json = json.dumps(energies)
with open(os.path.join(child_dir, f"{mol_name}_{str(global_id)}.json"), "a") as f:
    f.write(energies_json)
