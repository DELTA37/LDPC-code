from .base import BaseCode
import numpy as np
from typing import Tuple


class LinearCode(BaseCode):
    def __init__(self,
                 block_size,
                 code_size,
                 G=None,
                 H=None):
        super(LinearCode, self).__init__(block_size=block_size,
                                         code_size=code_size)

        if G is None and H is None:
            G = np.empty((block_size, code_size), dtype=np.int32)
            H = np.empty((code_size - block_size, code_size), dtype=np.int32)

        if H is None:
            P = G[:block_size, block_size:]  # block_size x (code_size - block_size)
            H = np.concatenate([P.T, np.eye(code_size - block_size, dtype=np.int32)], axis=1)  # (code_size - block_size) x code_size

        if G is None:
            P = H[:code_size - block_size, :block_size].T  # block_size x (code_size - block_size)
            G = np.concatenate([np.eye(block_size, dtype=np.int32), P], axis=1)

        self.G = G
        self.H = H
        print("G:")
        print(self.G)
        print("H:")
        print(self.H)

    def encode(self, array: np.ndarray) -> np.ndarray:
        return self.matmul(array, self.G)

    def decode(self, array: np.ndarray) -> np.ndarray:
        # TODO: table of error patterns
        raise NotImplementedError()

    def check_has_error(self, array: np.ndarray):
        e = self.matmul(self.H, array)
        print(e)
        return not np.all(e == np.zeros(self.code_size - self.block_size, dtype=np.int32))

    @staticmethod
    def bring_matrix_to_identity_residual_form(G: np.ndarray) -> Tuple[np.ndarray, bool]:
        # TODO: bring code to Cython
        for i in range(G.shape[0]):
            nz, = np.nonzero(G[i:, i])
            if nz.shape[0] == 0:
                y, x = np.nonzero(G[i:, i + 1:])
                if y.shape[0] == 0:
                    return G, False
                G[:, [i, i + 1 + x[0]]] = G[:, [i + 1 + x[0], i]]
            nz, = np.nonzero(G[i:, i])
            idx = i + nz[0]
            G[[i, idx]] = G[[idx, i]]
            nz = np.concatenate([np.nonzero(G[:i, i])[0], nz[1:]], axis=0)
            G[nz] = (G[nz] + G[i:i+1]) % 2
        return G, True
