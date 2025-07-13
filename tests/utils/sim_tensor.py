import functools
import itertools
import operator
from typing import Tuple

import numpy as np
import torch
from fontTools.misc.py23 import isclose
from hypothesis import strategies as st
from openfermion import count_qubits

from d_types.config_types import MConfig
from min_part.ham_utils import obtain_OF_hamiltonian


@st.composite
def symmetricND(draw, size: int) -> torch.Tensor:
    data = torch.zeros(*[size] * 4)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    data[i, j, k, l] = draw(st.integers(0, 1))
    return functools.reduce(
        operator.add,
        (
            torch.permute(data, permutation)
            for permutation in itertools.permutations(range(4))
        ),
    )


@st.composite
def artifical_h2_tbt(draw) -> np.ndarray:
    lmao0 = draw(st.floats(-1, 1))
    lmao1 = draw(st.floats(-1, 1))
    lmao2 = draw(st.floats(-1, 1))
    lmao3 = draw(st.floats(-1, 1))

    return np.array(
        [
            [
                [
                    [lmao0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao1 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao1 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j],
                    [lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ],
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [lmao0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao1 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao1 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j],
                    [lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ],
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j],
                    [lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [lmao1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao3 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao3 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ],
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao2 + 0.0j],
                    [lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [lmao1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, lmao1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, lmao3 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, lmao3 + 0.0j],
                ],
            ],
        ]
    )


def get_chem_tensors(H, N=None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Obtain constant, one-body tensor and two body tensor in chemist ordering from a FermionOperator in physicist or chemist ordering. Assumes the existance of 8 fold symmetry of molecular Hamiltonians.

    Args:
        H (FermionOperator): FermionOperator
        N (int, optional): Number of qubits. Defaults to None.

    Returns:
        float: Constant term in the Hamiltonian
        ndarray: One body tensor
        ndarray: Two body tensor in Chemist notation with all the appropriate (8) symmetries.
    """

    if N is None:
        N = count_qubits(H)

    for key, value in H.terms.items():
        if len(key) == 4:
            if key[1][1] == 1:  # Checking if the given term is physics ordered.
                obt = np.zeros((N, N), dtype="complex128")
                phy_tbt = np.zeros((N, N, N, N), dtype="complex128")
                const = 0
                obt_correction = np.zeros((N, N), dtype="complex128")

                for key, value in H.terms.items():
                    if len(key) == 4:
                        phy_tbt[key[0][0], key[1][0], key[2][0], key[3][0]] += value
                        if key[3][0] == key[1][0]:  # c0'c2'c1c2 = -c0'c2 + c0'c2c2'c1
                            obt_correction[key[0][0], key[2][0]] -= value
                    elif len(key) == 2:
                        obt[key[0][0], key[1][0]] += value
                    else:
                        const += value

                chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])
                chem_obt = obt + obt_correction
                return const, chem_obt, chem_tbt

            else:  # Assuming H is chemist ordered
                chem_obt = np.zeros((N, N), dtype="complex128")
                chem_tbt = np.zeros((N, N, N, N), dtype="complex128")
                const = 0
                for key, value in H.terms.items():
                    if len(key) == 4:
                        chem_tbt[key[0][0], key[1][0], key[2][0], key[3][0]] += (
                            value / 2
                        )
                        chem_tbt[key[2][0], key[3][0], key[0][0], key[1][0]] += (
                            value / 2
                        )
                    elif len(key) == 2:
                        chem_obt[key[0][0], key[1][0]] += value
                    else:
                        const += value
                return const, chem_obt, chem_tbt


def is_special_unitary(m):
    return (
        isclose(np.linalg.det(m), 1)
        and np.allclose(m @ m.T, np.identity(m.shape[0]))
        and np.allclose(m, m.T)
        and np.allclose(m, np.linalg.inv(m))
    )


def is_symmetric(m):
    return np.allclose(m, m.T)


def is_commute(q, l):
    assert l.shape == q.shape
    return np.allclose(q @ l, l @ q)


@st.composite
def generate_symm_unitary_matrices(draw, n):
    i = 0
    while True:
        i += 1
        A = np.array(
            [
                draw(st.lists(st.floats(-2, 2), min_size=n, max_size=n)),
                draw(st.lists(st.floats(-2, 2), min_size=n, max_size=n)),
                draw(st.lists(st.floats(-2, 2), min_size=n, max_size=n)),
                draw(st.lists(st.floats(-2, 2), min_size=n, max_size=n)),
            ]
        )
        A = (A + A.T) / 2
        eigenvals, eigenvecs = np.linalg.eigh(A)
        new_eigenvals = np.random.choice([-1, 1], size=n)
        maybe_u = eigenvecs @ np.diag(new_eigenvals) @ eigenvecs.T
        assert np.allclose(maybe_u, maybe_u.T)
        diags = np.array(
            draw(
                st.lists(
                    st.floats(-2, 2).filter(lambda x: not isclose(x, 0)),
                    min_size=n,
                    max_size=n,
                )
            )
        )
        maybe_symm = maybe_u @ np.diagflat(diags) @ maybe_u.T
        if (
            is_special_unitary(maybe_u)
            and is_symmetric(maybe_symm)
            and is_commute(maybe_u, np.diagflat(diags))
        ):
            break
    return diags, maybe_u, maybe_symm


def rand_symm_matr(n):
    i = 0
    while True:
        i += 1
        A = np.random.uniform(-2, 2, (n, n))
        A = (A + A.T) / 2
        eigenvals, eigenvecs = np.linalg.eigh(A)
        new_eigenvals = np.random.choice([-1, 1], size=n)
        maybe_u = eigenvecs @ np.diag(new_eigenvals) @ eigenvecs.T
        assert np.allclose(maybe_u, maybe_u.T)
        diags = np.random.uniform(-1, 1, 4)
        maybe_symm = maybe_u @ np.diagflat(diags) @ maybe_u.T
        if (
            is_special_unitary(maybe_u)
            and is_symmetric(maybe_symm)
            and is_commute(maybe_u, np.diagflat(diags))
        ):
            break
    return diags, maybe_u, maybe_symm


def make_tensors_h2(bond_length):
    mol = mol_h2(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


def get_tensors(
    m_config: MConfig, bond_length: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    mol = m_config.mol_of_interest(bond_length)
    H, num_elecs = obtain_OF_hamiltonian(mol)
    n_qubits = count_qubits(H)
    return get_chem_tensors(H=H, N=n_qubits)


from min_part.molecules import mol_h2  # noqa: E402
