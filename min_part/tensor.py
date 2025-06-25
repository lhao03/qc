from typing import Tuple, Optional

import numpy as np
import scipy as sp
from numpy import isclose
from openfermion import FermionOperator, count_qubits

from opt_einsum import contract

from d_types.config_types import Nums


def is_chemist_ordered(term: Tuple) -> bool:
    if len(term) % 2 != 0:
        return False
    creation_operator = True
    for _, op_type in term:
        if not creation_operator and bool(op_type):
            return False
        creation_operator = not creation_operator
    return True


def get_n_body_tensor_chemist_ordering(
    fo: FermionOperator, n: int, m: int
) -> np.ndarray:
    """Gets the rank 2^n tensor for FermionOperator, where the output tensor will be rank 2^n with dimension m,
     where m is the number of spin orbitals. Assumes the chemist ordering of operators.

    Args:
         n: refers to the `n`-body of the `FermionOperator`
         m: number of orbitals
         fo: an n-body tensor in chemist's ordering

    Raises:
        ValueError: if operator not in chemist ordering

    Returns:
        rank 2^n tensor with dimensions m
    """
    n = 2**n
    dimensions = [m for _ in range(n)]
    tensor = np.zeros(tuple(dimensions), dtype=np.float64)
    for term, coeff in fo.terms.items():
        if not is_chemist_ordered(term):
            raise ValueError(f"Expected chemist ordering, got: {term}")
        indices = [term[i][0] for i in range(n)]
        corr_pos = tensor
        for i in range(n - 1):
            corr_pos = corr_pos[indices[i]]
        corr_pos[indices[-1]] = coeff
    return tensor


def get_n_body_fo(tensor: np.ndarray) -> FermionOperator:
    """Creates the n-body `FermionOperator` from a rank log2(n) tensor"""
    n = tensor.shape[0]
    fo = FermionOperator()
    raise NotImplementedError


def get_no_from_tensor(lambda_m: np.ndarray) -> FermionOperator:
    """Creates the GFRO fragment in number operator form without orbital rotations.

    sum_{lm} lambda_{lm} n_l n_m

    Args:
        lambda_m: the n by n matrix defining the eigenvalues for the new orbitals from the U matrix. There are m unique values
        in lambda_m, where n * (n + 1) / 2 = m. n are the original number of spin orbitals (usually atomic ones).

    Returns:
        `FermionOperator` form of the fragment: sum_{lm} lambda_{lm} n_l n_m
    """
    s = lambda_m.shape[0]
    gfro_operator = FermionOperator()
    for l in range(s):
        for m in range(s):
            n_l = f"{str(l)}^ {str(l)}"
            n_m = f"{str(m)}^ {str(m)}"
            gfro_operator += FermionOperator(
                term=f"{n_l} {n_m}", coefficient=lambda_m[l][m]
            )
    return gfro_operator


def spin2spac(tensor):
    """
    Convert a one or two body tensor in spin orbital basis to spacial orbital basis assuming the existance of spin symmetry.

    Args:
        tensor (ndarray): one or two body tensor in spin orbital basis. Spin symmetry is assumed to exist.

    Returns:
        ndarray: corresponding one or two body tensor in spacial orbital basis
    """
    return tensor[np.ix_(*[range(0, i, 2) for i in tensor.shape])]


def spac2spin(tensor):
    """
    Convert a one or two body tensor in spacial orbital basis to spin orbital basis

    Args:
        tensor (ndarray): one or two body tensor in spacial orbital basis

    Returns:
        ndarray: corresponding one or two body tensor in spin orbital basis
    """
    N = tensor.shape[0]
    if len(tensor.shape) == 4:
        spin_tbt = np.zeros((2 * N, 2 * N, 2 * N, 2 * N), dtype="complex128")
        idx_uuuu = np.ix_(*[range(0, i, 2) for i in spin_tbt.shape])
        idx_dddd = np.ix_(*[range(1, i, 2) for i in spin_tbt.shape])
        idx_uudd = np.ix_(
            *[
                range(0, 2 * N, 2),
                range(0, 2 * N, 2),
                range(1, 2 * N, 2),
                range(1, 2 * N, 2),
            ]
        )
        idx_dduu = np.ix_(
            *[
                range(1, 2 * N, 2),
                range(1, 2 * N, 2),
                range(0, 2 * N, 2),
                range(0, 2 * N, 2),
            ]
        )
        spin_tbt[idx_uuuu] = spin_tbt[idx_dddd] = spin_tbt[idx_uudd] = spin_tbt[
            idx_dduu
        ] = tensor
        return spin_tbt
    elif len(tensor.shape) == 2:
        spin_obt = np.zeros((2 * N, 2 * N), dtype="complex128")
        idx_uu = np.ix_(*[range(0, i, 2) for i in spin_obt.shape])
        idx_dd = np.ix_(*[range(1, i, 2) for i in spin_obt.shape])
        spin_obt[idx_uu] = spin_obt[idx_dd] = tensor
        return spin_obt


def obtrot(obt, angles, N=None):
    """
    Given an obt, perform orbital rotation using "angles".

    Args:
        obt (ndarray): One body tensor
        angles (ndarray or list): Angles (theta_ij) defining the orbital rotation unitary, u. If u = exp(K), where K is anti Hermitian, K[i,j] = theta_ij, where ij is a composite index.

    Returns:
        ndarray: rotated obt.
    """
    if N == None:
        N = n_qubits

    kappa = np.zeros((N, N), dtype="complex128")
    idx = 0
    for p in range(N - 1):
        for q in range(p + 1, N):
            kappa[p, q] = angles[idx]
            kappa[q, p] = -angles[idx]
            idx += 1
    u = sp.linalg.expm(kappa)

    return contract("ij,pi,qj -> pq", obt, u, u)


def obt2tbt(obt):
    """
    Convert one-body-tensor to two-body-tensor in chemist ordering.

    Args:
        obt (ndarray): one-body tensor

    Returns:
        ndarray: two-body tensor
    """
    spac_obt = obt[
        np.ix_(*[range(0, i, 2) for i in obt.shape])
    ]  # Working with spacial orbitals: Extracting only one of the spin flavours
    N = spac_obt.shape[0]

    d, u = np.linalg.eigh(spac_obt)  # Assumes obt is Hermitian.

    spac_tbt = contract("i,pi,qi,ri,si->pqrs", d, u, u, u, u)

    spin_tbt = np.zeros((2 * N, 2 * N, 2 * N, 2 * N), dtype="complex128")
    idx_uuuu = np.ix_(*[range(0, i, 2) for i in spin_tbt.shape])
    idx_dddd = np.ix_(*[range(1, i, 2) for i in spin_tbt.shape])
    spin_tbt[idx_uuuu] = spin_tbt[idx_dddd] = spac_tbt

    return spin_tbt


def Bliss_K_tensors(
    ovec, t1, t2, η, n_qubits
):  # (n_qubits = n_qubits, len(ovec) = 2*(n_qubits/2)*(n_qubits/2 + 1)/2)
    """
    Build the tensor corresponding to Killer operator, K, which kills the states that are eigenvectors of number operator, N.
    K = t1(N - η) + t2(N^2 - η^2) + O(N - η). O is a one-body operator

    Args:
        ovec (ndarray or list): A vector of parameters defining the onebody operator O in Killer operator.
        t1 (float): Coefficient of (N - η)
        t2 (float): Coefficient of (N^2 - η^2)
        η (int): Number of electrons in the target subspace.
        n_qubits (int): Number of qubits.

    Returns:
        float: Constant in the Killer operator
        ndarray: One body tensor of Killer operator
        ndarray: Two body tensor of Killer operator
    """
    # builds S symmetry shift corresponding to S = s0+s1+s2

    obt = np.zeros((n_qubits, n_qubits))
    idx = 0
    for i in range(int(n_qubits / 2)):
        for j in range(i + 1):
            obt[2 * i, 2 * j] = ovec[idx]
            obt[2 * j, 2 * i] = ovec[idx]
            obt[2 * i + 1, 2 * j + 1] = ovec[idx + 1]
            obt[2 * j + 1, 2 * i + 1] = ovec[idx + 1]
            idx += 2

    s0 = -t1 * (η**2) - t2 * η
    s1 = t2 * np.diag(np.ones(n_qubits)) - η * obt

    s2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(n_qubits):
            s2[i, i, j, j] += t1
            for k in range(n_qubits):
                s2[i, j, k, k] += 0.5 * obt[i, j]
                s2[k, k, i, j] += 0.5 * obt[i, j]
    return s0, s1, s2


def obt2op(obt):
    """
    convert one-body-tensor to FermionOperator

    Args:
        obt (np.array): one-body-tensor

    Returns:
        FermionOperator: FermionOperator corresponding to the input one-body-tensor
    """
    op = FermionOperator()
    N = obt.shape[0]
    for p in range(N):
        for q in range(N):
            term = ((p, 1), (q, 0))
            coef = obt[p, q]
            op += FermionOperator(term, float(coef))
    return op


def tbt2op(tbt):
    """
    convert chemist two-body-tensor to FermionOperator

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        FermionOperator: FermionOperator corresponding to the input chemist ordered two-body-tensor
    """
    op = FermionOperator()
    N = tbt.shape[0]
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    term = ((p, 1), (q, 0), (r, 1), (s, 0))
                    coef = tbt[p, q, r, s]
                    op += FermionOperator(term, float(coef))
    return op


def fbt2op(fbt):
    """
    Convert chemist four-body-tensor to FermionOperator

    Args:
        fbt (np.array): four-body-tensor

    Returns:
        FermionOperator: FermionOperator corresponding to the input chemist ordered four-body-tensor
    """
    op = FermionOperator()
    N = fbt.shape[0]
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    for a in range(N):
                        for b in range(N):
                            for c in range(N):
                                for d in range(N):
                                    term = (
                                        (p, 1),
                                        (q, 0),
                                        (r, 1),
                                        (s, 0),
                                        (a, 1),
                                        (b, 0),
                                        (c, 1),
                                        (d, 0),
                                    )
                                    coef = fbt[p, q, r, s, a, b, c, d]
                                    op += FermionOperator(term, coef)
    return op


def get_chem_tensors_hubbard(H_chem, N=None):  # Does not work in general.
    """
    Not sure, but specifically meant for Hubbard model obtained from OpenFermion.
    """
    # H_chem = chemist_ordered(H)

    if N == None:
        N = count_qubits(H_chem)

    chem_obt = np.zeros((N, N), dtype="complex128")
    chem_tbt = np.zeros((N, N, N, N), dtype="complex128")
    const = 0

    for key, value in H_chem.terms.items():
        if len(key) == 4:
            p, q, r, s = key[0][0], key[1][0], key[2][0], key[3][0]

            if (
                np.abs(p - q) % 2 != 0
            ):  # If p-q is not even, it means p and q are of different spin flavour
                q, s = (
                    s,
                    q,
                )  # Interchange q with s so that p&s and r&q are now of the same spin flavour.
                value = -value

            # Enforcing all the 8 tensor symmetries
            chem_tbt[p, q, r, s] += value / 8
            chem_tbt[q, p, r, s] += value / 8
            chem_tbt[p, q, s, r] += value / 8
            chem_tbt[q, p, s, r] += value / 8
            chem_tbt[r, s, p, q] += value / 8
            chem_tbt[s, r, p, q] += value / 8
            chem_tbt[r, s, q, p] += value / 8
            chem_tbt[s, r, q, p] += value / 8

        elif len(key) == 2:
            chem_obt[key[0][0], key[1][0]] += value / 2
            chem_obt[key[1][0], key[0][0]] += value / 2
        else:
            const += value

    return const, chem_obt, chem_tbt


def make_x_matrix(
    thetas: np.ndarray, n: int, diags: Optional[np.ndarray] = None, imag: bool = False
) -> np.ndarray:
    """Makes the X matrix required to define a unitary orbital rotation, where

    U = e^X

    Elements are filled into X starting at a diagonal element at (i, i) and then filling the ith column and ith row.

    So given an N by N matrix, we use n elements in theta for the 1st row and column, then (n-1) elements for the 2nd
    row and column, etc...

    Args:
        thetas: angles required to make the X matrix, need N(N+1)/2 angles
        n: used for the dimension of the X matrix

    Returns:
        an N by N matrix
    """
    expected_num_angles = (n * (n + 1) // 2) - n
    if thetas.size != expected_num_angles:
        raise UserWarning(
            f"Expected {expected_num_angles} angles for a {n} by {n} X matrix, got {thetas.size}."
        )
    if imag:
        if not isinstance(diags, np.ndarray):
            raise UserWarning(
                "Since the X matrix might be imaginary, there might be diagonal elements."
            )
    X = np.zeros((n, n), dtype=np.complex128 if imag else np.float64)
    t = 0
    for x in range(n):
        for y in range(x + 1, n):
            val = thetas[t]
            v_real = val.real
            v_imag = val.imag
            if not isclose(0, v_real) and not isclose(0, v_imag):
                X[x][y] = complex(real=-v_real, imag=v_imag)
                X[y][x] = complex(real=v_real, imag=v_imag)
            elif not isclose(0, v_real) and isclose(0, v_imag):
                X[x][y] = -v_real
                X[y][x] = v_real
            elif isclose(0, v_real) and not isclose(0, v_imag):
                X[x][y] = complex(real=0, imag=v_imag)
                X[y][x] = complex(real=0, imag=v_imag)
            else:
                pass
            t += 1
    if imag:
        for i, d in enumerate(diags):
            d_real = d
            d_imag = 0
            if isinstance(d, complex):
                d_real = d.real
                d_imag = d.imag
            r_is_0 = isclose(d_real, 0)
            im_is_0 = isclose(d_imag, 0)
            if r_is_0 and im_is_0:
                X[i, i] = 0
            elif r_is_0:
                X[i, i] = complex(0, d_imag)
            elif im_is_0:
                X[i, i] = d_real
            else:
                X[i, i] = d
    return X


def make_unitary(thetas: Nums, n: int, imag: bool = False) -> np.ndarray:
    X = make_x_matrix(np.array(thetas), n, imag=imag)
    u = sp.linalg.expm(X)
    u.setflags(write=True)
    num_err = np.finfo(u.dtype).eps
    tol = num_err * 10
    u.real[abs(u.real) < tol] = 0.0
    return u


def make_lambda_matrix(lambdas: np.ndarray, n: int) -> np.ndarray:
    expected_size = n * (n + 1) // 2
    if lambdas.size != expected_size:
        raise UserWarning(
            f"Expected {expected_size} angles for a {n} by {n} lambda matrix, got {lambdas.size}."
        )
    l = np.random.rand(n, n)
    t = 0
    for x in range(n):
        for y in range(x, n):
            l[y][x] = lambdas[t]
            l[x][y] = lambdas[t]
            t += 1
    return l


def extract_thetas(U) -> Tuple[Nums, Nums]:
    """Extracts theta values from a unitary matrix paramertized by real amplitudes.
    Args:
        U: the unitary

    Returns:
        theta values
    """
    X: np.ndarray = sp.linalg.logm(U)
    m = ((U.shape[0] * (U.shape[0] + 1)) // 2) - U.shape[0]
    thetas = np.zeros((m,), dtype=np.complex128)
    u = U.shape[0]
    counter = 0
    for i in range(u - 1):
        for j in range(i + 1, u):
            thetas[counter] = X[j, i]
            counter += 1
    return thetas, X.diagonal()


def make_unitary_im(thetas, diags, n):
    X = make_x_matrix(np.array(thetas), n, diags=diags, imag=True)
    return sp.linalg.expm(X)


def extract_lambdas(c_matrix, n):
    m = (n * (n + 1)) // 2
    lambdas = np.zeros((m,))
    c = 0
    for i in range(c_matrix.shape[0]):
        for j in range(i, c_matrix.shape[0]):
            lambdas[c] = c_matrix[i, j]
            c += 1
    return lambdas
