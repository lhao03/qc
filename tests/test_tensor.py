import unittest

import numpy as np
from hypothesis import given
from opt_einsum import contract

from min_part.tensor import generate_symm_unitary_matrices


class TensorTest(unittest.TestCase):
    @given(generate_symm_unitary_matrices(n=4))
    def test_uni_decomp(self, u_symm):
        diags = u_symm[0]
        u = u_symm[1]
        symm = u_symm[2]
        self.assertEqual(np.linalg.det(u), 1)
        vals, vecs = np.linalg.eigh(symm)
        np.testing.assert_array_equal(vecs @ np.diagflat(vals) @ vecs.T, vecs.T @ np.diagflat(vals) @ vecs)
        np.testing.assert_array_equal(u @ np.diagflat(diags) @ u.T, u.T @ np.diagflat(diags) @ u)
        self.assertEqual(vals.size, 4)
        self.assertEqual(diags.size, 4)
        np.testing.assert_array_equal(
            contract("r,pr,qr->pq", vals, vecs, vecs),
            contract("r,qr,pr->pq", vals, vecs, vecs),
        )
        np.testing.assert_array_equal(
            contract("r,rp,rq->pq", vals, vecs, vecs),
            contract("r,pr,qr->pq", vals, vecs, vecs),
        )
        np.testing.assert_array_equal(
            contract("r,rp,rq->pq", diags, u, u),
            contract("r,pr,qr->pq", diags, u, u),
        )
        np.testing.assert_array_equal(
            contract("r,rp,rq->pq", diags, u, u),
            contract("r,rp,rq->pq", vals, vecs, vecs),
        )
