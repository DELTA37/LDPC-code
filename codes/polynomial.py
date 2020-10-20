from .linear import LinearCode
from .poly_gf2 import poly1d_gf2
import numpy as np


class PolynomialCode(LinearCode):
    """
    """
    def __init__(self, block_size, q: poly1d_gf2):
        self.q = q
        self.degree = q.shape[0] - 1
        self.block_size = block_size
        self.code_size = block_size + self.degree
        self.raw_G = self.construct_generator_matrix_with_polynom(q, systematic=False)
        G, status = self.bring_matrix_to_identity_residual_form(self.raw_G)
        assert status
        super(PolynomialCode, self).__init__(self.block_size, self.code_size,
                                             G=G)

    def construct_generator_matrix_with_polynom(self, q: np.ndarray,
                                                systematic=False) -> np.ndarray:
        """
        q[0] + q[1] * x + ... q[r] x ** r
        :param q:
        :return:
        """
        assert q.shape[0] == self.degree + 1
        G = np.zeros(([self.block_size, self.code_size]), dtype=np.int32)
        for i in range(self.block_size):
            G[i, i:i + q.shape[0]] = q
        if not systematic:
            return G
        G = self.bring_matrix_to_identity_residual_form(G)
        assert G[1]
        return G[0]

    def poly_divide(self, a, b):
        a = self.poly_strip_zeros(a)
        b = self.poly_strip_zeros(b)
        if a.shape[0] < b.shape[0]:
            return np.array([], dtype=np.int32), a
        q = self.poly_create(a.shape[0] - b.shape[0])
        r = self.poly_add(self.poly_mul(q, b), a)

    def poly_add(self, a, b):
        p_a = np.poly1d(np.flipud(a))
        p_b = np.poly1d(np.flipud(b))
        return self.poly_strip_zeros(np.flipud((p_a + p_b).c))

    def poly_degree(self, c):
        c = self.poly_strip_zeros(c)
        return c.shape[0] - 1

    def poly_mul(self, a, b):
        p_a = np.poly1d(np.flipud(a))
        p_b = np.poly1d(np.flipud(b))
        return self.poly_strip_zeros(np.flipud((p_a * p_b).c))

    def poly_strip_zeros(self, a):
        return np.flipud(np.poly1d(np.flipud(a % 2)).c)

    def poly_create(self, order):
        c = np.zeros((order + 1), dtype=np.int32)
        c[-1] = 1
        return c
