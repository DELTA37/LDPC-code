from .linear import LinearCode
import numpy as np
import bitarray


class HammingCode(LinearCode):
    def __init__(self, r):
        powers = [2 ** i for i in range(r)]
        P = np.stack([self.int2binary(i, r) for i in range(1, 2 ** r) if i not in powers], axis=-1)  # r x 2**r-1
        H = np.concatenate([
            P,
            np.eye(r, dtype=np.int32),
        ], axis=-1)
        super(HammingCode, self).__init__(block_size=2 ** r - 1 - r,
                                          code_size=2 ** r - 1,
                                          H=H)

    def decode(self, array: np.ndarray) -> np.ndarray:
        array = array.copy()
        e = self.matmul(self.H, array)
        if not np.all(e == np.zeros(self.code_size - self.block_size, dtype=np.int32)):
            idx = np.argmax(np.all(e.reshape((1, self.code_size - self.block_size)) == self.H.T, axis=-1))
            array[idx] = (array[idx] + 1) % 2
        return array[:self.block_size]
