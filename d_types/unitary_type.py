from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from numpy import isclose

from d_types.config_types import Nums
from julia import Pkg

Pkg.activate("/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/MolHamLinAlg")
from julia import MolHamLinAlg


def check_complexity_and_zero(m):
    tol = 1e-8
    m.real[abs(m.real) < tol] = 0.0
    m.imag[abs(m.imag) < tol] = 0.0
    return np.real_if_close(m, tol=10000000)


@dataclass
class Unitary:
    dim: int

    @abstractmethod
    def make_unitary_matrix(self):
        raise NotImplementedError

    @classmethod
    def deconstruct_unitary(cls, u: np.ndarray):
        dim = u.shape[0]
        try:
            thetas, diags_thetas = jl_extract_thetas(u)
            thetas_filtered = check_complexity_and_zero(thetas)
            diags_filtered = check_complexity_and_zero(diags_thetas)
            if np.all(diags_filtered == 0):
                return ReaDeconUnitary(thetas=thetas_filtered, dim=dim)
            else:
                return ComDeconUnitary(
                    thetas=thetas_filtered, diag_thetas=diags_filtered, dim=dim
                )
        except Exception:
            raise UserWarning("Failed to deconstruct unitary, storing it entirely.")


@dataclass
class ReaDeconUnitary(Unitary):
    thetas: np.ndarray

    def make_unitary_matrix(self):
        X = make_x_matrix(np.array(self.thetas), self.dim, imag=False)
        u = sp.linalg.expm(X)
        u.setflags(write=True)
        num_err = np.finfo(u.dtype).eps
        tol = num_err * 10
        u.real[abs(u.real) < tol] = 0.0
        return u


@dataclass
class ComDeconUnitary(Unitary):
    thetas: np.ndarray
    diag_thetas: np.ndarray

    def make_unitary_matrix(self):
        X = make_x_matrix(
            np.array(self.thetas), self.dim, diags=self.diag_thetas, imag=True
        )
        return sp.linalg.expm(X)


@dataclass
class WholeUnitary(Unitary):
    mat: np.ndarray

    def make_unitary_matrix(self):
        return self.mat


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


def jl_extract_thetas(u) -> tuple[np.ndarray, np.ndarray]:
    return MolHamLinAlg.extract_thetas(u)


def jl_make_x_im(t, d, n) -> np.ndarray:
    return MolHamLinAlg.make_x_matrix(t, d, n)


def jl_make_x(t, n) -> np.ndarray:
    return MolHamLinAlg.make_x_matrix(t, n)


def jl_make_u_im(t, d, n) -> np.ndarray:
    return MolHamLinAlg.make_unitary(t, d, n)


def jl_make_u(t, d) -> np.ndarray:
    return MolHamLinAlg.make_unitary(t, d)
