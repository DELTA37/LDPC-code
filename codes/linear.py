from .base import BaseCode
import numpy as np


class LinearCode(BaseCode):
    def __init__(self, block_size, code_size):
        super(LinearCode, self).__init__(block_size=block_size,
                                         code_size=code_size)

        self.G = np.zeros((block_size, code_size), dtype=np.int32)
        self.H = np.zeros((code_size - block_size, code_size), dtype=np.int32)

    def encode(self, array: np.ndarray) -> np.ndarray:
        return array.dot(self.G)

    def decode(self, array: np.ndarray) -> np.ndarray:
        return array.dot(self.H)
