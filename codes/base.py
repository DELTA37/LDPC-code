import os
import numpy as np
import math
import bitarray
from typing import Union, List


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
    def int2binary(x, zfill_num):
        x = np.array(bitarray.bitarray(bin(x)[2:].zfill(zfill_num)).tolist(), dtype=np.int32)
        return x

    @staticmethod
    def matmul(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a.dot(x) % 2

    @staticmethod
    def flip(x: np.ndarray) -> np.ndarray:
        return (x + 1) % 2

    @staticmethod
    def flip_bits(x: np.ndarray, bit_idx: Union[int, List[int]]) -> np.ndarray:
        if not isinstance(bit_idx, list):
            bit_idx = [bit_idx]
        x = x.copy()
        x[bit_idx] = (x[bit_idx] + 1) % 2
        return x

    @staticmethod
    def hamming(a, b):
        return np.sum((a + b) % 2)

    def iter_codewords(self):
        for i in range(2 ** self.block_size):
            block = np.array(bitarray.bitarray(bin(i)[2:].zfill(self.block_size)).tolist(), dtype=np.int32)
            code = self.encode(block)
            yield block, code

    def __repr__(self):
        return "Code(%s, %s)" % (self.code_size, self.block_size)
