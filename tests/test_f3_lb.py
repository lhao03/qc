import json
import os
import unittest
from typing import Tuple

import numpy as np
from openfermion import count_qubits

from d_types.config_types import MConfig
from d_types.fragment_types import Subspace, PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.f3_optimis import subspace_operators
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import h2_settings
from min_part.plots import RefLBPlotNames, plot_energies, FluidPlotNames
from tests.utils.sim_tensor import get_chem_tensors


def get_tensors(
    m_config: MConfig, bond_length: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    mol = m_config.mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


class F3Test(unittest.TestCase):
    # == saving nums
    no_partitioning = []
    lr = []
    lr_f3 = []
    gfro = []
    gfro_f3 = []

    def test_partition(self, bond_length: float, m_config: MConfig):
        number_operator, sz, s2 = subspace_operators(m_config)
        const, obt, tbt = get_tensors(m_config, bond_length)
        reference = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        lr = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        E = reference.get_expectation_value()
        gfro_frags = gfro.partition(
            strategy=PartitionStrategy.GFRO, bond_length=bond_length
        )
        lr_frags = lr.partition(strategy=PartitionStrategy.LR, bond_length=bond_length)
        self.assertEqual(
            sum([f.operators for f in gfro_frags]), sum([f.operators for f in lr_frags])
        )
        E_gfro = gfro.get_expectation_value()
        E_lr = lr.get_expectation_value()
        self.assertTrue(E >= E_gfro)
        self.assertTrue(E >= E_lr)
        return E, E_gfro, E_lr

    def test_make_lb(self):
        child_dir = os.path.join(
            "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/data/h2",
            h2_settings.date,
        )
        no_partitioning = []
        gfro = []
        lr = []
        for bond_length in h2_settings.xpoints:
            print(f"Partitioning: {bond_length} A")
            E, E_gfro, E_lr = self.test_partition(bond_length, m_config=h2_settings)
            no_partitioning.append(E)
            gfro.append(E_gfro)
            lr.append(E_lr)
        plot_energies(
            xpoints=h2_settings.xpoints,
            points=[no_partitioning, gfro, lr],
            title=f"{h2_settings.mol_name} Lower Bounds",
            labels=[
                RefLBPlotNames.NO_PARTITIONING,
                RefLBPlotNames.LR_N_S,
                RefLBPlotNames.GFRO_N_S,
            ],
            dir=child_dir,
        )
        energies = {
            RefLBPlotNames.NO_PARTITIONING.value: no_partitioning,
            RefLBPlotNames.LR_N_S.value: lr,
            RefLBPlotNames.GFRO_N_S.value: gfro,
        }

        energies_json = json.dumps(energies)
        with open(
            os.path.join(child_dir, f"{h2_settings.mol_name}.json"),
            "w",
        ) as f:
            f.write(energies_json)

    def test_simple_functionality(self, bond_length: float = 1, m_config=h2_settings):
        number_operator, sz, s2 = subspace_operators(m_config)
        const, obt, tbt = get_tensors(m_config, bond_length)
        self.assertTrue(isinstance(const, float))
        self.assertEqual(obt.shape, (m_config.num_spin_orbs, m_config.num_spin_orbs))
        self.assertEqual(
            tbt.shape,
            (
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
                m_config.num_spin_orbs,
            ),
        )
        reference = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        E = reference.get_expectation_value()
        self.assertTrue(-2 <= E <= 0)
        saved_location = reference.save()
        loaded_reference = FragmentedHamiltonian.load(saved_location)
        self.assertEqual(reference, loaded_reference)

    def test_gfro_opt(self, bond_length=0.8, m_config=h2_settings):
        print(f"Partitioning bond {bond_length}")
        number_operator, sz, s2 = subspace_operators(m_config)
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(number_operator, 2, s2, 0, sz, 0),
        )
        E = gfro.get_expectation_value()
        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        gfro_E = gfro.get_expectation_value()
        gfro.optimize_fragments()
        gfro_fluid_E = gfro.get_expectation_value()
        gfro.save(id=str(bond_length))
        # self.assertTrue(E >= gfro_fluid_E >= gfro_E)
        return E, gfro_fluid_E, gfro_E

    def test_fluid_gfro(self):
        child_dir = os.path.join(
            "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/data/h2",
            h2_settings.date,
        )
        no_partitioning = []
        gfro_fluid = []
        gfro = []
        for bond_length in h2_settings.xpoints:
            E, gfro_fluid_E, gfro_E = self.test_gfro_opt(
                bond_length=bond_length, m_config=h2_settings
            )
            print(f"E: {E}, Fluid GFRO E: {gfro_fluid_E}, GFRO E: {gfro_E}")
            no_partitioning.append(E)
            gfro_fluid.append(gfro_fluid_E)
            gfro.append(gfro_E)
        plot_energies(
            xpoints=h2_settings.xpoints,
            points=[no_partitioning, gfro_fluid, gfro],
            title=f"GFRO Lower Bounds for {h2_settings.mol_name}",
            labels=[
                FluidPlotNames.NO_PARTITIONING,
                FluidPlotNames.GFRO_FLUID,
                FluidPlotNames.GFRO,
            ],
            dir=child_dir,
        )
        energies = {
            FluidPlotNames.NO_PARTITIONING.value: no_partitioning,
            FluidPlotNames.GFRO_FLUID.value: gfro_fluid,
            FluidPlotNames.GFRO.value: gfro,
        }

        energies_json = json.dumps(energies)
        with open(
            os.path.join(child_dir, f"{h2_settings.mol_name}.json"),
            "w",
        ) as f:
            f.write(energies_json)
