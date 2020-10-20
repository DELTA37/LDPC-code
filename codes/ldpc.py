import os
from .linear import LinearCode
import numpy as np


class LDPCCode(LinearCode):
    def __init__(self, code_size,
                 d_v=2, d_c=4):
        H = self.create_parity_check_matrix(code_size, d_v, d_c)

        Href_colonnes, tQ = self.gaussjordan(H.T, 1)
        Href_diag = self.gaussjordan(np.transpose(Href_colonnes))
        Q = tQ.T
        n_bits = code_size - Href_diag.sum()
        Y = np.zeros(shape=(code_size, n_bits)).astype(int)
        Y[code_size - n_bits:, :] = np.identity(n_bits)
        G = self.binaryproduct(Q, Y)

        self.d_v = d_v
        self.d_c = d_c
        super(LDPCCode, self).__init__(block_size=n_bits, code_size=code_size,
                                       H=H, G=G)

    @staticmethod
    def create_parity_check_matrix(code_size: int, d_v: int, d_c: int):
        """
        Callager algo
        :param code_size:
        :param d_v:
        :param d_c:
        :return:
        """
        assert d_v > 1
        assert d_c > d_v
        assert code_size % d_c == 0

        n_equations = (code_size * d_v) // d_c
        block = np.zeros((n_equations // d_v, code_size), dtype=int)
        H = np.empty((n_equations, code_size), dtype=np.int32)
        block_size = n_equations // d_v

        for i in range(block_size):
            for j in range(i * d_c, (i + 1) * d_c):
                block[i, j] = 1
        H[:block_size] = block

        for i in range(1, d_v):
            H[i * block_size: (i + 1) * block_size] = np.random.permutation(block.T).T
        return H

    def encode(self, array: np.ndarray) -> np.ndarray:
        return self.binaryproduct(self.G, array)
