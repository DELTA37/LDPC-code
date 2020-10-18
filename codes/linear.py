from .base import BaseCode
import numpy as np


class LinearCode(BaseCode):
    def __init__(self,
                 block_size,
                 code_size,
                 G=None,
                 H=None):
        super(LinearCode, self).__init__(block_size=block_size,
                                         code_size=code_size)

        if G is None and H is None:
            G = np.zeros((block_size, code_size), dtype=np.int32)
            H = np.zeros((code_size - block_size, code_size), dtype=np.int32)

        if H is None:
            P = G[:block_size, block_size:]  # block_size x (code_size - block_size)
            H = np.concatenate([P.T, np.eye(code_size - block_size)], axis=1)  # (code_size - block_size) x code_size

        if G is None:
            P = H[:code_size - block_size, :block_size].T  # block_size x (code_size - block_size)
            G = np.concatenate([P, np.eye(block_size)], axis=1)

        self.G = G
        self.H = H

    def encode(self, array: np.ndarray) -> np.ndarray:
        return array.dot(self.G)

    def decode(self, array: np.ndarray) -> np.ndarray:
        return array.dot(self.H)
