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

    def encode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    def string2bit(self, s: str) -> np.ndarray:
        assert BaseCode.is_ascii(s)
        x = bitarray.bitarray(''.join([bin(ord(x))[2:].zfill(self.block_size) for x in s]))
        return np.array(x.tolist(), dtype=np.int32)

    def bit2string(self, x: np.ndarray) -> str:
        assert x.shape[0] % self.block_size == 0
        x = ''.join(map(str, (x % 2).tolist()))
        x = ''.join([chr(int(x[i: i + self.block_size], 2)) for i in range(0, len(x), self.block_size)])
        return x

    @staticmethod
    def matmul(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a.dot(x) % 2

    @staticmethod
    def flip(x: np.ndarray) -> np.ndarray:
        return (x + 1) % 2

    @staticmethod
    def hamming(a, b):
        return np.sum((a + b) % 2)
