from .linear import LinearCode
import numpy as np


class PolynomialCode(LinearCode):
    def __init__(self, block_size, q):
        code_size = block_size + q.shape[0] - 1
        self.degree = q.shape[0] - 1
        self.block_size = block_size
        self.code_size = code_size
        G = self.construct_generator_matrix_with_polynom(q)
        super(PolynomialCode, self).__init__(block_size, code_size,
                                             G=G)

    def construct_generator_matrix_with_polynom(self, q: np.ndarray) -> np.ndarray:
        """
        q[0] + q[1] * x + ... q[r] x ** r
        :param q:
        :return:
        """
        assert q.shape[0] == self.degree + 1
        G = np.zeros(([self.block_size, self.code_size]), dtype=np.int32)
        for i in range(self.block_size):
            G[i, i:i + q.shape[0]] = q
        G = self.bring_matrix_to_identity_residual_form(G)
        assert G[1]
        return G[0]
