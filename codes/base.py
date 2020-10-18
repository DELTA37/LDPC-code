import os
import numpy as np
import math
import bitarray


class BaseCode(object):
    block_size: int

    def __init__(self, block_size: int = 7, code_size: int = 3):
        super(BaseCode, self).__init__()
        self.block_size = block_size
        self.code_size = code_size

    @property
    def rate(self):
        return self.block_size / self.code_size

    def encode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


    @staticmethod
    def matmul(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a.dot(x) % 2

    @staticmethod
    def flip(x: np.ndarray) -> np.ndarray:
        return (x + 1) % 2

    @staticmethod
    def hamming(a, b):
        return np.sum((a + b) % 2)

    def __repr__(self):
        return "Code(%s, %s)" % (self.code_size, self.block_size)
