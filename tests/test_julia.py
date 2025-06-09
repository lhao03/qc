import unittest

import numpy as np

from min_part.operators import extract_eigenvalue


class JuliaTest(unittest.TestCase):
    def test_julia_call(self):
        from julia import Pkg
        Pkg.activate("/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/julia/MolHamLinAlg")
        from julia import MolHamLinAlg
        import scipy as sp
        op = sp.sparse.csr_matrix(np.ones((2, 2)))
        vec = np.array([1,1])
        jl_eig = MolHamLinAlg.extract_eigen(np.ones((2,2)), vec)
        py_eig = extract_eigenvalue(op, vec)
        self.assertTrue(np.array_equal(jl_eig * vec, py_eig * vec))
