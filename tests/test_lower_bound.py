import os
import random
import unittest
from functools import reduce

import numpy as np
import scipy as sp
from hypothesis import settings
from openfermion import (
    count_qubits,
    jordan_wigner,
    FermionOperator,
)
from openfermion.linalg import qubit_operator_sparse

from min_part.gfro_decomp import make_unitary, gfro_decomp
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.lr_decomp import lr_decomp
from min_part.molecules import h2_settings
from min_part.tensor import obt2op, tbt2op, get_chem_tensors
from min_part.utils import (
    diag_partitioned_fragments,
    save_frags,
    save_energies,
    load_energies,
    get_saved_file_names,
)
from min_part.plots import plot_energies, PlotNames

settings.register_profile("slow", deadline=None)
settings.load_profile("slow")


class LowerBoundTest(unittest.TestCase):
    def test_h2_lb(self):
        global gfro_files
        no_partitioning = []

        lr_n_subspace_energies = []
        lr_n_s_subspace_energies = []
        lr_all_subspace_energies = []

        gfro_n_subspace_energies = []
        gfro_n_s_subspace_energies = []
        gfro_all_subspace_energies = []

        config_settings = h2_settings
        global_id = str(random.randint(0, 100))
        parent_dir = f"../data/{config_settings.mol_name.lower()}"
        child_dir = os.path.join(parent_dir, config_settings.date, str(global_id))

        load = False
        continue_on = False
        if load or continue_on:
            gfro_files, lr_files = get_saved_file_names(child_dir)
            (
                no_partitioning,
                lr_n_subspace_energies,
                gfro_n_subspace_energies,
                lr_n_s_subspace_energies,
                gfro_n_s_subspace_energies,
                lr_all_subspace_energies,
                gfro_all_subspace_energies,
            ) = load_energies(child_dir, config_settings, global_id)
        elif not continue_on:
            while os.path.exists(child_dir):
                global_id = random.randint(0, 100)
                child_dir = os.path.join(
                    parent_dir, config_settings.date, str(global_id)
                )
            if not os.path.isdir(child_dir):
                os.makedirs(os.path.join(child_dir, "gfro"))
                os.makedirs(os.path.join(child_dir, "lr"))

        for i, bond_length in enumerate(config_settings.xpoints):
            print(f"Now partitioning: {bond_length} angstroms.")
            mol = config_settings.mol_of_interest(bond_length)
            H, num_elecs = obtain_OF_hamiltonian(mol)
            n_qubits = count_qubits(H)
            H_const, H_obt, H_tbt = get_chem_tensors(H=H, N=n_qubits)
            H_ob_op = obt2op(H_obt)
            H_tb_op = tbt2op(H_tbt)
            H_ele = H_const + H_ob_op + H_tb_op
            eigenvalues, eigenvectors = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(H_ele)).toarray()
            )
            unpartitioned_energy = eigenvalues[0]
            no_partitioning.append(unpartitioned_energy)

            gfro_data = gfro_decomp(
                tbt=H_tbt, previous_thetas=None, previous_lambdas=None
            )
            lr_data = lr_decomp(tbt=H_tbt)
            lr_frags_jl = [f.operators for f in lr_data]
            gfro_frags = [f.operators for f in gfro_data]

            H_no_two_body = H_const * FermionOperator.identity() + H_ob_op
            h1_v, h1_w = sp.linalg.eigh(
                qubit_operator_sparse(jordan_wigner(H_no_two_body)).toarray()
            )

            lr_operator_sum_jl = reduce(lambda op1, op2: op1 + op2, lr_frags_jl)
            gfro_operator_sum = reduce(lambda op1, op2: op1 + op2, gfro_frags)
            self.assertEqual(lr_operator_sum_jl, H_tb_op)
            self.assertEqual(gfro_operator_sum, H_tb_op)

            for gr_data, lr_data in zip(gfro_data, lr_data):
                u = make_unitary(gr_data.thetas, config_settings.num_spin_orbs)
                self.assertTrue(np.isclose(np.linalg.det(u), 1))

            print("LR Fragment Analysis")
            lr_n_subspace_energy, lr_n_s_energy, lr_all_subspace_energy = (
                diag_partitioned_fragments(
                    h2_frags=lr_frags_jl,
                    h1_v=h1_v,
                    h1_w=h1_w,
                    num_elecs=num_elecs,
                    num_spin_orbs=config_settings.num_spin_orbs,
                )
            )
            lr_n_subspace_energies.append(lr_n_subspace_energy)
            lr_n_s_subspace_energies.append(lr_n_s_energy)
            lr_all_subspace_energies.append(lr_all_subspace_energy)

            print("GFRO Fragment Analysis")
            (
                gfro_n_subspace_energy,
                gfro_n_s_subspace_energy,
                gfro_all_subspace_energy,
            ) = diag_partitioned_fragments(
                h2_frags=gfro_data,
                h1_v=h1_v,
                h1_w=h1_w,
                num_elecs=num_elecs,
                num_spin_orbs=config_settings.num_spin_orbs,
            )
            gfro_n_subspace_energies.append(gfro_n_subspace_energy)
            gfro_n_s_subspace_energies.append(gfro_n_s_subspace_energy)
            gfro_all_subspace_energies.append(gfro_all_subspace_energy)

            if not load:
                save_frags(gfro_data, os.path.join(child_dir, "gfro", str(i)))
                save_frags(lr_frags_jl, os.path.join(child_dir, "lr", str(i)))
                save_energies(
                    child_dir,
                    config_settings,
                    global_id,
                    no_partitioning,
                    lr_n_subspace_energies,
                    gfro_n_subspace_energies,
                    lr_n_s_subspace_energies,
                    gfro_n_s_subspace_energies,
                    lr_all_subspace_energies,
                    gfro_all_subspace_energies,
                )

        plot_energies(
            xpoints=config_settings.xpoints,
            points=[
                no_partitioning,
                lr_n_subspace_energies,
                gfro_n_subspace_energies,
            ],
            title=f"{config_settings.mol_name} Energies from Partitioning, No Spin Constraints",
            labels=[PlotNames.NO_PARTITIONING, PlotNames.LR_N, PlotNames.GFRO_N],
            dir=child_dir,
        )

        plot_energies(
            xpoints=config_settings.xpoints,
            points=[
                no_partitioning,
                lr_n_s_subspace_energies,
                gfro_n_s_subspace_energies,
            ],
            title=f"{config_settings.mol_name} Energies from Partitioning, Spin Constraints",
            labels=[PlotNames.NO_PARTITIONING, PlotNames.LR_N_S, PlotNames.GFRO_N_S],
            dir=child_dir,
        )

        plot_energies(
            xpoints=config_settings.xpoints,
            points=[
                no_partitioning,
                lr_n_subspace_energies,
                gfro_n_subspace_energies,
                lr_n_s_subspace_energies,
                gfro_n_s_subspace_energies,
                lr_all_subspace_energies,
                gfro_all_subspace_energies,
            ],
            title=f"{config_settings.mol_name} Energies from Partitioning, All Bounds",
            labels=[
                PlotNames.NO_PARTITIONING,
                PlotNames.LR_N,
                PlotNames.GFRO_N,
                PlotNames.LR_N_S,
                PlotNames.GFRO_N_S,
                PlotNames.LR_F_SPACE,
                PlotNames.GFRO_F_SPACE,
            ],
            dir=child_dir,
        )

        plot_energies(
            xpoints=config_settings.xpoints,
            points=[
                no_partitioning,
                lr_n_subspace_energies,
                gfro_n_subspace_energies,
                lr_n_s_subspace_energies,
                gfro_n_s_subspace_energies,
            ],
            title=f"{config_settings.mol_name} Energies from Partitioning, Tightest Bounds",
            labels=[
                PlotNames.NO_PARTITIONING,
                PlotNames.LR_N,
                PlotNames.GFRO_N,
                PlotNames.LR_N_S,
                PlotNames.GFRO_N_S,
            ],
            dir=child_dir,
        )
