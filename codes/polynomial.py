from .linear import LinearCode
from .poly_gf2 import poly1d_gf2
import numpy as np


class PolynomialCode(LinearCode):
    """
    https://web.ntpu.edu.tw/~yshan/cyclic_code.pdf
    """
    def __init__(self, block_size, q: poly1d_gf2):
        self.q = q
        self.block_size = block_size
        self.code_size = block_size + q.order
        h, _ = self.h, residual = (poly1d_gf2.create_basis(self.code_size) + poly1d_gf2.create_basis(0)).euclid_div(self.q)
        print(f"q: {repr(self.q)}")
        print(f"h: {repr(self.h)}")
        assert not residual
        self.raw_G = self.construct_generator_matrix_with_polynom(q, systematic=False)
        self.raw_H = self.construct_check_matrix_with_polynom(h.flipud())
        G, status = self.bring_matrix_to_identity_residual_form(self.raw_G.copy())
        assert status
        print("raw_G:")
        print(self.raw_G)
        print("raw_H:")
        print(self.raw_H)
        super(PolynomialCode, self).__init__(self.block_size, self.code_size,
                                             G=G)

    def construct_generator_matrix_with_polynom(self, q: poly1d_gf2,
                                                systematic: bool = False) -> np.ndarray:
        """
        :param q:
        :param systematic:
        :return:
        """
        q = np.flipud(q.coeffs)
        G = np.zeros(([self.block_size, self.code_size]), dtype=np.int32)
        for i in range(self.block_size):
            G[i, i:i + q.shape[0]] = q
        if not systematic:
            return G
        G = self.bring_matrix_to_identity_residual_form(G)
        assert G[1]
        return G[0]

    def construct_check_matrix_with_polynom(self, h: poly1d_gf2) -> np.ndarray:
        """
        :param h:
        :param systematic:
        :return:
        """
        q = np.flipud(h.coeffs)
        H = np.zeros(([self.code_size - self.block_size, self.code_size]), dtype=np.int32)
        for i in range(self.code_size - self.block_size):
            H[i, i:i + q.shape[0]] = q
        return H

    def encode(self, array: np.ndarray) -> np.ndarray:
        u = poly1d_gf2.from_code(array) * poly1d_gf2.create_basis(self.code_size - self.block_size)
        _, b = u.euclid_div(self.q)
        return (u + b).to_code(zfill=self.code_size)

    def decode(self, array: np.ndarray) -> np.ndarray:
        code = poly1d_gf2.from_code(array)
        _, s = code.euclid_div(self.q)
        if not s:
            return (code // poly1d_gf2.create_basis(self.q.order)).to_code(zfill=self.block_size)
        return (code // poly1d_gf2.create_basis(self.q.order)).to_code(zfill=self.block_size)
