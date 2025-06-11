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
        mat_rand = np.random.rand(100,100)
        U, V = self.mhla.eigendecomp(mat_rand)
        U, V = np.array(U), np.array(V)
        julia_res = U @ V @ np.linalg.inv(U)
        self.assertTrue(np.allclose(julia_res, mat_rand))

