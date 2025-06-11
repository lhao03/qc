import unittest

import numpy as np

from min_part.operators import extract_eigenvalue


class JuliaTest(unittest.TestCase):
    def setUp(self):
        from julia import Pkg

        Pkg.activate(
            "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/julia/MolHamLinAlg"
        )
        from julia import MolHamLinAlg

        self.mhla = MolHamLinAlg

    def test_julia_call(self):
        import scipy as sp

        op = sp.sparse.csr_matrix(np.ones((2, 2)))
        vec = np.array([1, 1])
        jl_eig = self.mhla.extract_eigen(np.ones((2, 2)), vec, True)
        py_eig = extract_eigenvalue(op, vec)
        self.assertTrue(np.array_equal(jl_eig * vec, py_eig * vec))

    def test_julia_eigendecomp(self):
        import time

        a = np.random.rand(1000, 1000)
        mat_rand = np.tril(a) + np.tril(a, -1).T

        t0 = time.time()
        U, V = self.mhla.UV_eigendecomp(mat_rand)
        U, V = np.array(U), np.array(V)
        julia_res = U @ np.diagflat(V) @ np.linalg.inv(U)
        t1 = time.time()
        print(t1 - t0)

        t0 = time.time()
        np_V, np_U = np.linalg.eigh(mat_rand)
        np_res = np_U @ np.diagflat(np_V) @ np.linalg.inv(np_U)
        t1 = time.time()
        print(t1 - t0)

        self.assertTrue(np.allclose(julia_res, mat_rand))
        self.assertTrue(np.allclose(np_res, mat_rand))
        self.assertTrue(np.allclose(np_res, julia_res))
