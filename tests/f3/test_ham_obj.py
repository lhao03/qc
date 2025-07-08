import json
import os
import unittest
from typing import Tuple

import numpy as np
from openfermion import count_qubits, jordan_wigner, qubit_operator_sparse

from d_types.config_types import MConfig
from d_types.fragment_types import Subspace, PartitionStrategy
from d_types.hamiltonian import FragmentedHamiltonian
from min_part.ham_utils import obtain_OF_hamiltonian
from min_part.molecules import h2_settings
from min_part.plots import RefLBPlotNames, plot_energies
from min_part.tensor import obt2op, tbt2op
from tests.utils.sim_tensor import get_chem_tensors


def get_tensors(
    m_config: MConfig, bond_length: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    mol = m_config.mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


def create_ham_objs(const, m_config, obt, tbt):
    reference = FragmentedHamiltonian(
        m_config=m_config,
        constant=const,
        one_body=obt,
        two_body=tbt,
        partitioned=False,
        fluid=False,
        subspace=Subspace(expected_e=2, expected_sz=0, expected_s2=0),
    )
    gfro = FragmentedHamiltonian(
        m_config=m_config,
        constant=const,
        one_body=obt,
        two_body=tbt,
        partitioned=False,
        fluid=False,
        subspace=Subspace(expected_e=2, expected_sz=0, expected_s2=0),
    )
    lr = FragmentedHamiltonian(
        m_config=m_config,
        constant=const,
        one_body=obt,
        two_body=tbt,
        partitioned=False,
        fluid=False,
        subspace=Subspace(expected_e=2, expected_sz=0, expected_s2=0),
    )
    return gfro, lr, reference


class HamTest(unittest.TestCase):
    def test_partition(self, bond_length: float = 0.8, m_config: MConfig = h2_settings):
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro, lr, reference = create_ham_objs(const, m_config, obt, tbt)
        E = reference.get_expectation_value()
        gfro_frags = gfro.partition(
            strategy=PartitionStrategy.GFRO, bond_length=bond_length
        )
        lr_frags = lr.partition(strategy=PartitionStrategy.LR, bond_length=bond_length)
        self.assertEqual(
            sum([f.operators for f in gfro_frags]), sum([f.operators for f in lr_frags])
        )
        proj_E_lr = lr.get_expectation_value(use_frag_energies=False)
        E_lr = lr.get_expectation_value(
            use_frag_energies=True, desired_occs=[(0, 1), (1, 2), (1, 3), (2, 3)]
        )
        proj_E_gfro = gfro.get_expectation_value(use_frag_energies=False)
        E_gfro = gfro.get_expectation_value(
            use_frag_energies=True, desired_occs=[(0, 1), (1, 2), (1, 3), (2, 3)]
        )
        self.assertAlmostEqual(E_gfro, proj_E_gfro)
        self.assertAlmostEqual(E_lr, proj_E_lr)
        self.assertTrue(E >= E_gfro)
        self.assertTrue(E >= E_lr)
        return E, E_gfro, E_lr

    def test_make_lb(self):
        child_dir = os.path.join(
            "/data/h2",
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
                RefLBPlotNames.GFRO_N_S,
                RefLBPlotNames.LR_N_S,
            ],
            dir=child_dir,
        )
        energies = {
            RefLBPlotNames.NO_PARTITIONING.value: no_partitioning,
            RefLBPlotNames.GFRO_N_S.value: gfro,
            RefLBPlotNames.LR_N_S.value: lr,
        }

        energies_json = json.dumps(energies)
        with open(
            os.path.join(child_dir, f"{h2_settings.mol_name}.json"),
            "w",
        ) as f:
            f.write(energies_json)

    def test_simple_functionality(self, bond_length: float = 1, m_config=h2_settings):
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
            subspace=Subspace(2, 0, 0),
        )
        E = reference.get_expectation_value()
        self.assertTrue(-2 <= E <= 0)
        saved_location = reference.save()
        loaded_reference = FragmentedHamiltonian.load(saved_location)
        self.assertEqual(reference, loaded_reference)

    def test_manual_f3_gfro_proj(self, bond_length=0.8, m_config=h2_settings):
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        print(f"OG Energy: {gfro.get_expectation_value()}")
        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        print(f"Partitioned Energy: {gfro.get_expectation_value()}")
        original_operator_sum = gfro.get_operators()
        for j, f in enumerate(gfro.two_body):
            f.to_fluid()
            for i, c in enumerate(f.fluid_parts.fluid_lambdas):
                if i == 3 or i == 1 or i == 0:
                    f.move2frag(gfro.one_body, -c, i, mutate=True)
                    print(f"Moved {-c} from orb {i}, E: {gfro.get_expectation_value()}")
                    try:
                        self.assertNotEqual(original_operator_sum, gfro.get_operators())
                        self.assertEqual(
                            jordan_wigner(original_operator_sum),
                            jordan_wigner(gfro.get_operators()),
                        )
                        self.assertAlmostEqual(
                            gfro._trace(original_operator_sum),
                            gfro._trace(gfro.get_operators()),
                        )
                    except:
                        print("Failed JW check")
        self.assertEqual(
            jordan_wigner(original_operator_sum), jordan_wigner(gfro.get_operators())
        )
        print(f"Final Energy: {gfro.get_expectation_value()}")

    def test_manual_f3_lr(self, bond_length=0.8, m_config=h2_settings):
        const, obt, tbt = get_tensors(m_config, bond_length)
        lr = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        print(f"OG Energy: {lr.get_expectation_value()}")
        lr.partition(strategy=PartitionStrategy.LR, bond_length=bond_length)
        original_operator_sum = lr.get_operators()
        print(f"Partitioned Energy: {lr.get_expectation_value()}")
        for j, f in enumerate(lr.two_body):
            f.to_fluid()
            for i, c in enumerate(f.fluid_parts.fluid_lambdas):
                f.move2frag(lr.one_body, -c * 10, i, mutate=True)
                print(
                    f"Energy after moving {-c * 10} to orb {i}: {lr.get_expectation_value()}"
                )
                try:
                    self.assertNotEqual(original_operator_sum, lr.get_operators())
                    self.assertEqual(
                        jordan_wigner(original_operator_sum),
                        jordan_wigner(lr.get_operators()),
                    )
                    self.assertAlmostEqual(
                        lr._trace(original_operator_sum),
                        lr._trace(lr.get_operators()),
                    )
                except:
                    print("Failed JW check")
        self.assertEqual(
            jordan_wigner(original_operator_sum), jordan_wigner(lr.get_operators())
        )
        print(f"Final Energy: {lr.get_expectation_value()}")

    def test_diagonalization_strats_and_weyls(
        self, bond_length=0.8, m_config=h2_settings
    ):
        const, obt, tbt = get_tensors(m_config, bond_length)
        reference, lr, gfro = create_ham_objs(const, m_config, obt, tbt)
        ref_E_ss = reference.get_expectation_value()
        ref_E_compl = reference._diagonalize_operator_complete_ss(
            reference.constant + obt2op(reference.one_body) + tbt2op(reference.two_body)
        )
        self.assertAlmostEqual(ref_E_ss, ref_E_compl)

        lr.partition(strategy=PartitionStrategy.LR, bond_length=bond_length)
        lr_E_ss = lr.get_expectation_value()
        lr_E_compl = lr._diagonalize_operator_complete_ss(
            lr.constant + lr.one_body.to_op()
        ) + sum(
            [lr._diagonalize_operator_complete_ss(f.operators) for f in lr.two_body]
        )
        lr_E_orb_occ = lr._diagonalize_operator_with_ss_proj(
            lr.constant + lr.one_body.to_op()
        ) + sum([lr._add_up_orb_occs(f) for f in lr.two_body])
        self.assertNotEqual(lr_E_ss, lr_E_compl)  # LR Partitioning Shows this Bug
        self.assertAlmostEqual(lr_E_ss, lr_E_orb_occ)

        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        gfro_E_ss = gfro.get_expectation_value()
        gfro_E_compl = gfro._diagonalize_operator_complete_ss(
            gfro.constant + gfro.one_body.to_op()
        ) + sum(
            [gfro._diagonalize_operator_complete_ss(f.operators) for f in gfro.two_body]
        )
        gfro_E_occs = gfro._diagonalize_operator_with_ss_proj(
            gfro.constant + gfro.one_body.to_op()
        ) + sum([gfro._add_up_orb_occs(f) for f in gfro.two_body])
        self.assertAlmostEqual(
            gfro_E_ss, gfro_E_compl
        )  # GFRO Partitioning Does Not Show Bug
        self.assertAlmostEqual(gfro_E_ss, gfro_E_occs)

    def test_f3_gfro_violate_weyls(self, bond_length=0.8, m_config=h2_settings):
        const, obt, tbt = get_tensors(m_config, bond_length)
        gfro = FragmentedHamiltonian(
            m_config=m_config,
            constant=const,
            one_body=obt,
            two_body=tbt,
            partitioned=False,
            fluid=False,
            subspace=Subspace(2, 0, 0),
        )
        vals, vecs = np.linalg.eigh(
            qubit_operator_sparse(
                jordan_wigner(
                    gfro.constant + obt2op(gfro.one_body) + tbt2op(gfro.two_body)
                )
            ).toarray()
        )
        print(f"OG Energy: {gfro.get_expectation_value()}")
        gfro.partition(strategy=PartitionStrategy.GFRO, bond_length=bond_length)
        original_operator_sum = gfro.get_operators()
        for j, f in enumerate(gfro.two_body):
            gfro.two_body[j] = f.to_fluid()
            for i, c in enumerate(f.fluid_parts.fluid_lambdas):
                f.move2frag(gfro.one_body, c, i, mutate=True)
                ob_e = gfro._diagonalize_operator_complete_ss(
                    gfro.constant + gfro.one_body.to_op()
                )
                eigs = sum(
                    [
                        gfro._diagonalize_operator_complete_ss(f.operators)
                        for f in gfro.two_body
                    ]
                )
                fluid_sum = ob_e + eigs
                print("===")
                print(f"Moved {c} from orb {i}.")
                for i, v in enumerate(reversed(vals)):
                    if fluid_sum <= v:
                        print(f"Index {i}, where {fluid_sum} <= {v}")
                print("===")
                try:
                    self.assertNotEqual(original_operator_sum, gfro.get_operators())
                    self.assertEqual(
                        jordan_wigner(original_operator_sum),
                        jordan_wigner(gfro.get_operators()),
                    )
                    self.assertAlmostEqual(
                        gfro._diagonalize_operator_with_ss_proj(original_operator_sum),
                        gfro._diagonalize_operator_with_ss_proj(gfro.get_operators()),
                    )
                except:
                    print("Failed JW check")
        self.assertEqual(
            jordan_wigner(original_operator_sum), jordan_wigner(gfro.get_operators())
        )
        print(f"Final Energy: {gfro.get_expectation_value()}")
