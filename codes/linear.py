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
        array = array.copy()
        e = self.matmul(self.H, array)
        if not np.all(e == np.zeros(self.code_size - self.block_size, dtype=np.int32)):
            idx = np.argmax(np.all(e.reshape((1, self.code_size - self.block_size)) == self.H.T, axis=-1))
            array[idx] = (array[idx] + 1) % 2
        return array[:self.block_size]
